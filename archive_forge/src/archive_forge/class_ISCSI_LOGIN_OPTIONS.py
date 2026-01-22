import ctypes
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
class ISCSI_LOGIN_OPTIONS(ctypes.Structure):
    _fields_ = [('Version', wintypes.ULONG), ('InformationSpecified', ISCSI_LOGIN_OPTIONS_INFO_SPECIFIED), ('LoginFlags', ISCSI_LOGIN_FLAGS), ('AuthType', ISCSI_AUTH_TYPES), ('HeaderDigest', ISCSI_DIGEST_TYPES), ('DataDigest', ISCSI_DIGEST_TYPES), ('MaximumConnections', wintypes.ULONG), ('DefaultTime2Wait', wintypes.ULONG), ('DefaultTime2Retain', wintypes.ULONG), ('UsernameLength', wintypes.ULONG), ('PasswordLength', wintypes.ULONG), ('Username', wintypes.PSTR), ('Password', wintypes.PSTR)]