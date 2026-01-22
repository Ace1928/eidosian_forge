import ctypes
import functools
import inspect
import socket
import time
from oslo_log import log as logging
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
class ISCSIInitiatorUtils(object):
    _DEFAULT_RESCAN_ATTEMPTS = 3
    _MS_IQN_PREFIX = 'iqn.1991-05.com.microsoft'
    _DEFAULT_ISCSI_PORT = 3260

    def __init__(self):
        self._win32utils = win32utils.Win32Utils()
        self._diskutils = diskutils.DiskUtils()

    def _run_and_check_output(self, *args, **kwargs):
        kwargs['error_msg_src'] = iscsierr.err_msg_dict
        kwargs['failure_exc'] = exceptions.ISCSIInitiatorAPIException
        self._win32utils.run_and_check_output(*args, **kwargs)

    @ensure_buff_and_retrieve_items(struct_type=iscsi_struct.PERSISTENT_ISCSI_LOGIN_INFO)
    def _get_iscsi_persistent_logins(self, buff=None, buff_size=None, element_count=None):
        self._run_and_check_output(iscsidsc.ReportIScsiPersistentLoginsW, ctypes.byref(element_count), buff, ctypes.byref(buff_size))

    @ensure_buff_and_retrieve_items(struct_type=ctypes.c_wchar, func_requests_buff_sz=False, parse_output=False)
    def get_targets(self, forced_update=False, buff=None, buff_size=None, element_count=None):
        """Get the list of iSCSI targets seen by the initiator service."""
        self._run_and_check_output(iscsidsc.ReportIScsiTargetsW, forced_update, ctypes.byref(element_count), buff)
        return self._parse_string_list(buff, element_count.value)

    def get_iscsi_initiator(self):
        """Returns the initiator node name."""
        try:
            buff = (ctypes.c_wchar * (w_const.MAX_ISCSI_NAME_LEN + 1))()
            self._run_and_check_output(iscsidsc.GetIScsiInitiatorNodeNameW, buff)
            return buff.value
        except exceptions.ISCSIInitiatorAPIException as ex:
            LOG.info("The ISCSI initiator node name can't be found. Choosing the default one. Exception: %s", ex)
            return '%s:%s' % (self._MS_IQN_PREFIX, socket.getfqdn().lower())

    @ensure_buff_and_retrieve_items(struct_type=ctypes.c_wchar, func_requests_buff_sz=False, parse_output=False)
    def get_iscsi_initiators(self, buff=None, buff_size=None, element_count=None):
        """Get the list of available iSCSI initiator HBAs."""
        self._run_and_check_output(iscsidsc.ReportIScsiInitiatorListW, ctypes.byref(element_count), buff)
        return self._parse_string_list(buff, element_count.value)

    @staticmethod
    def _parse_string_list(buff, element_count):
        buff = ctypes.cast(buff, ctypes.POINTER(ctypes.c_wchar))
        str_list = buff[:element_count].strip('\x00')
        str_list = str_list.split('\x00') if str_list else []
        return str_list

    @retry_decorator(error_codes=w_const.ERROR_INSUFFICIENT_BUFFER)
    def _login_iscsi_target(self, target_name, portal=None, login_opts=None, is_persistent=True, initiator_name=None):
        session_id = iscsi_struct.ISCSI_UNIQUE_SESSION_ID()
        connection_id = iscsi_struct.ISCSI_UNIQUE_CONNECTION_ID()
        portal_ref = ctypes.byref(portal) if portal else None
        login_opts_ref = ctypes.byref(login_opts) if login_opts else None
        initiator_name_ref = ctypes.c_wchar_p(initiator_name) if initiator_name else None
        self._run_and_check_output(iscsidsc.LoginIScsiTargetW, ctypes.c_wchar_p(target_name), False, initiator_name_ref, ctypes.c_ulong(w_const.ISCSI_ANY_INITIATOR_PORT), portal_ref, iscsi_struct.ISCSI_SECURITY_FLAGS(), None, login_opts_ref, ctypes.c_ulong(0), None, is_persistent, ctypes.byref(session_id), ctypes.byref(connection_id), ignored_error_codes=[w_const.ISDSC_TARGET_ALREADY_LOGGED_IN])
        return (session_id, connection_id)

    @ensure_buff_and_retrieve_items(struct_type=iscsi_struct.ISCSI_SESSION_INFO)
    def _get_iscsi_sessions(self, buff=None, buff_size=None, element_count=None):
        self._run_and_check_output(iscsidsc.GetIScsiSessionListW, ctypes.byref(buff_size), ctypes.byref(element_count), buff)

    def _get_iscsi_target_sessions(self, target_name, connected_only=True):
        sessions = self._get_iscsi_sessions()
        return [session for session in sessions if session.TargetNodeName and session.TargetNodeName.upper() == target_name.upper() and (session.ConnectionCount > 0 or not connected_only)]

    @retry_decorator(error_codes=(w_const.ISDSC_SESSION_BUSY, w_const.ISDSC_DEVICE_BUSY_ON_SESSION))
    @ensure_buff_and_retrieve_items(struct_type=iscsi_struct.ISCSI_DEVICE_ON_SESSION, func_requests_buff_sz=False)
    def _get_iscsi_session_devices(self, session_id, buff=None, buff_size=None, element_count=None):
        self._run_and_check_output(iscsidsc.GetDevicesForIScsiSessionW, ctypes.byref(session_id), ctypes.byref(element_count), buff)

    def _get_iscsi_session_disk_luns(self, session_id):
        devices = self._get_iscsi_session_devices(session_id)
        luns = [device.ScsiAddress.Lun for device in devices if device.StorageDeviceNumber.DeviceType == w_const.FILE_DEVICE_DISK]
        return luns

    def _get_iscsi_device_from_session(self, session_id, target_lun):
        devices = self._get_iscsi_session_devices(session_id)
        for device in devices:
            if device.ScsiAddress.Lun == target_lun:
                return device

    def get_device_number_for_target(self, target_name, target_lun, fail_if_not_found=False):
        return self.get_device_number_and_path(target_name, target_lun, fail_if_not_found)[0]

    def get_device_number_and_path(self, target_name, target_lun, fail_if_not_found=False, retry_attempts=10, retry_interval=0.1, rescan_disks=False, ensure_mpio_claimed=False):
        device_number, device_path = (None, None)
        try:
            device_number, device_path = self.ensure_lun_available(target_name, target_lun, rescan_attempts=retry_attempts, retry_interval=retry_interval, rescan_disks=rescan_disks, ensure_mpio_claimed=ensure_mpio_claimed)
        except exceptions.ISCSILunNotAvailable:
            if fail_if_not_found:
                raise
        return (device_number, device_path)

    def get_target_luns(self, target_name):
        sessions = self._get_iscsi_target_sessions(target_name)
        if sessions:
            luns = self._get_iscsi_session_disk_luns(sessions[0].SessionId)
            return luns
        return []

    def get_target_lun_count(self, target_name):
        return len(self.get_target_luns(target_name))

    @retry_decorator(error_codes=w_const.ISDSC_SESSION_BUSY)
    def _logout_iscsi_target(self, session_id):
        self._run_and_check_output(iscsidsc.LogoutIScsiTarget, ctypes.byref(session_id))

    def _add_static_target(self, target_name, is_persistent=True):
        self._run_and_check_output(iscsidsc.AddIScsiStaticTargetW, ctypes.c_wchar_p(target_name), None, 0, is_persistent, None, None, None)

    def _remove_static_target(self, target_name):
        ignored_error_codes = [w_const.ISDSC_TARGET_NOT_FOUND]
        self._run_and_check_output(iscsidsc.RemoveIScsiStaticTargetW, ctypes.c_wchar_p(target_name), ignored_error_codes=ignored_error_codes)

    def _get_login_opts(self, auth_username=None, auth_password=None, auth_type=None, login_flags=0):
        if auth_type is None:
            auth_type = constants.ISCSI_CHAP_AUTH_TYPE if auth_username and auth_password else constants.ISCSI_NO_AUTH_TYPE
        login_opts = iscsi_struct.ISCSI_LOGIN_OPTIONS()
        info_bitmap = 0
        if auth_username:
            login_opts.Username = six.b(auth_username)
            login_opts.UsernameLength = len(auth_username)
            info_bitmap |= w_const.ISCSI_LOGIN_OPTIONS_USERNAME
        if auth_password:
            login_opts.Password = six.b(auth_password)
            login_opts.PasswordLength = len(auth_password)
            info_bitmap |= w_const.ISCSI_LOGIN_OPTIONS_PASSWORD
        login_opts.AuthType = auth_type
        info_bitmap |= w_const.ISCSI_LOGIN_OPTIONS_AUTH_TYPE
        login_opts.InformationSpecified = info_bitmap
        login_opts.LoginFlags = login_flags
        return login_opts

    def _session_on_path_exists(self, target_sessions, portal_addr, portal_port, initiator_name):
        for session in target_sessions:
            connections = session.Connections[:session.ConnectionCount]
            uses_requested_initiator = False
            if initiator_name:
                devices = self._get_iscsi_session_devices(session.SessionId)
                for device in devices:
                    if device.InitiatorName == initiator_name:
                        uses_requested_initiator = True
                        break
            else:
                uses_requested_initiator = True
            for conn in connections:
                is_requested_path = uses_requested_initiator and conn.TargetAddress == portal_addr and (conn.TargetSocket == portal_port)
                if is_requested_path:
                    return True
        return False

    def _new_session_required(self, target_iqn, portal_addr, portal_port, initiator_name, mpio_enabled):
        login_required = False
        sessions = self._get_iscsi_target_sessions(target_iqn)
        if not sessions:
            login_required = True
        elif mpio_enabled:
            login_required = not self._session_on_path_exists(sessions, portal_addr, portal_port, initiator_name)
        return login_required

    def login_storage_target(self, target_lun, target_iqn, target_portal, auth_username=None, auth_password=None, auth_type=None, mpio_enabled=False, ensure_lun_available=True, initiator_name=None, rescan_attempts=_DEFAULT_RESCAN_ATTEMPTS):
        portal_addr, portal_port = _utils.parse_server_string(target_portal)
        portal_port = int(portal_port) if portal_port else self._DEFAULT_ISCSI_PORT
        known_targets = self.get_targets()
        if target_iqn not in known_targets:
            self._add_static_target(target_iqn)
        login_required = self._new_session_required(target_iqn, portal_addr, portal_port, initiator_name, mpio_enabled)
        if login_required:
            LOG.debug('Logging in iSCSI target %(target_iqn)s', dict(target_iqn=target_iqn))
            login_flags = w_const.ISCSI_LOGIN_FLAG_MULTIPATH_ENABLED if mpio_enabled else 0
            login_opts = self._get_login_opts(auth_username, auth_password, auth_type, login_flags)
            portal = iscsi_struct.ISCSI_TARGET_PORTAL(Address=portal_addr, Socket=portal_port)
            self._login_iscsi_target(target_iqn, portal, login_opts, is_persistent=True)
            sid, cid = self._login_iscsi_target(target_iqn, portal, login_opts, is_persistent=False)
        if ensure_lun_available:
            self.ensure_lun_available(target_iqn, target_lun, rescan_attempts)

    def ensure_lun_available(self, target_iqn, target_lun, rescan_attempts=_DEFAULT_RESCAN_ATTEMPTS, retry_interval=0, rescan_disks=True, ensure_mpio_claimed=False):
        for attempt in range(rescan_attempts + 1):
            sessions = self._get_iscsi_target_sessions(target_iqn)
            for session in sessions:
                try:
                    sid = session.SessionId
                    device = self._get_iscsi_device_from_session(sid, target_lun)
                    if not device:
                        continue
                    device_number = device.StorageDeviceNumber.DeviceNumber
                    device_path = device.LegacyName
                    if not device_path or device_number in (None, -1):
                        continue
                    if ensure_mpio_claimed and (not self._diskutils.is_mpio_disk(device_number)):
                        LOG.debug('Disk %s was not claimed yet by the MPIO service.', device_path)
                        continue
                    return (device_number, device_path)
                except exceptions.ISCSIInitiatorAPIException:
                    err_msg = 'Could not find lun %(target_lun)s  for iSCSI target %(target_iqn)s.'
                    LOG.exception(err_msg, dict(target_lun=target_lun, target_iqn=target_iqn))
                    continue
            if attempt <= rescan_attempts:
                if retry_interval:
                    time.sleep(retry_interval)
                if rescan_disks:
                    self._diskutils.rescan_disks()
        raise exceptions.ISCSILunNotAvailable(target_lun=target_lun, target_iqn=target_iqn)

    @retry_decorator(error_codes=(w_const.ISDSC_SESSION_BUSY, w_const.ISDSC_DEVICE_BUSY_ON_SESSION))
    def logout_storage_target(self, target_iqn):
        LOG.debug('Logging out iSCSI target %(target_iqn)s', dict(target_iqn=target_iqn))
        sessions = self._get_iscsi_target_sessions(target_iqn, connected_only=False)
        for session in sessions:
            self._logout_iscsi_target(session.SessionId)
        self._remove_target_persistent_logins(target_iqn)
        self._remove_static_target(target_iqn)

    def _remove_target_persistent_logins(self, target_iqn):
        persistent_logins = self._get_iscsi_persistent_logins()
        for persistent_login in persistent_logins:
            if persistent_login.TargetName == target_iqn:
                LOG.debug('Removing iSCSI target persistent login: %(target_iqn)s', dict(target_iqn=target_iqn))
                self._remove_persistent_login(persistent_login)

    def _remove_persistent_login(self, persistent_login):
        self._run_and_check_output(iscsidsc.RemoveIScsiPersistentTargetW, ctypes.c_wchar_p(persistent_login.InitiatorInstance), persistent_login.InitiatorPortNumber, ctypes.c_wchar_p(persistent_login.TargetName), ctypes.byref(persistent_login.TargetPortal))