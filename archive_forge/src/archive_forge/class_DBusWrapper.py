from __future__ import absolute_import, division, print_function
import os
import sys
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.respawn import (
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils import deps
class DBusWrapper(object):
    """
    Helper class that can be used for running a command with a working D-Bus
    session.

    If possible, command will be run against an existing D-Bus session,
    otherwise the session will be spawned via dbus-run-session.

    Example usage:

    dbus_wrapper = DBusWrapper(ansible_module)
    dbus_wrapper.run_command(["printenv", "DBUS_SESSION_BUS_ADDRESS"])
    """

    def __init__(self, module):
        """
        Initialises an instance of the class.

        :param module: Ansible module instance used to signal failures and run commands.
        :type module: AnsibleModule
        """
        self.module = module
        self.dbus_session_bus_address = self._get_existing_dbus_session()
        if self.dbus_session_bus_address is None:
            self.dbus_run_session_cmd = self.module.get_bin_path('dbus-run-session', required=True)

    def _get_existing_dbus_session(self):
        """
        Detects and returns an existing D-Bus session bus address.

        :returns: string -- D-Bus session bus address. If a running D-Bus session was not detected, returns None.
        """
        uid = os.getuid()
        self.module.debug('Trying to detect existing D-Bus user session for user: %d' % uid)
        for pid in psutil.pids():
            try:
                process = psutil.Process(pid)
                process_real_uid, dummy, dummy = process.uids()
                if process_real_uid == uid and 'DBUS_SESSION_BUS_ADDRESS' in process.environ():
                    dbus_session_bus_address_candidate = process.environ()['DBUS_SESSION_BUS_ADDRESS']
                    self.module.debug('Found D-Bus user session candidate at address: %s' % dbus_session_bus_address_candidate)
                    dbus_send_cmd = self.module.get_bin_path('dbus-send', required=True)
                    command = [dbus_send_cmd, '--address=%s' % dbus_session_bus_address_candidate, '--type=signal', '/', 'com.example.test']
                    rc, dummy, dummy = self.module.run_command(command)
                    if rc == 0:
                        self.module.debug('Verified D-Bus user session candidate as usable at address: %s' % dbus_session_bus_address_candidate)
                        return dbus_session_bus_address_candidate
            except psutil.AccessDenied:
                pass
            except psutil.NoSuchProcess:
                pass
        self.module.debug('Failed to find running D-Bus user session, will use dbus-run-session')
        return None

    def run_command(self, command):
        """
        Runs the specified command within a functional D-Bus session. Command is
        effectively passed-on to AnsibleModule.run_command() method, with
        modification for using dbus-run-session if necessary.

        :param command: Command to run, including parameters. Each element of the list should be a string.
        :type module: list

        :returns: tuple(result_code, standard_output, standard_error) -- Result code, standard output, and standard error from running the command.
        """
        if self.dbus_session_bus_address is None:
            self.module.debug('Using dbus-run-session wrapper for running commands.')
            command = [self.dbus_run_session_cmd] + command
            rc, out, err = self.module.run_command(command)
            if self.dbus_session_bus_address is None and rc == 127:
                self.module.fail_json(msg='Failed to run passed-in command, dbus-run-session faced an internal error: %s' % err)
        else:
            extra_environment = {'DBUS_SESSION_BUS_ADDRESS': self.dbus_session_bus_address}
            rc, out, err = self.module.run_command(command, environ_update=extra_environment)
        return (rc, out, err)