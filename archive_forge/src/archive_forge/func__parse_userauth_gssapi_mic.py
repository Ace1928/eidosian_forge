import weakref
import threading
import time
import re
from paramiko.common import (
from paramiko.message import Message
from paramiko.util import b, u
from paramiko.ssh_exception import (
from paramiko.server import InteractiveQuery
from paramiko.ssh_gss import GSSAuth, GSS_EXCEPTIONS
def _parse_userauth_gssapi_mic(self, m):
    mic_token = m.get_string()
    sshgss = self.sshgss
    username = self.auth_username
    self._restore_delegate_auth_handler()
    try:
        sshgss.ssh_check_mic(mic_token, self.transport.session_id, username)
    except Exception as e:
        self.transport.saved_exception = e
        result = AUTH_FAILED
        self._send_auth_result(username, self.method, result)
        raise
    result = AUTH_SUCCESSFUL
    self.transport.server_object.check_auth_gssapi_with_mic(username, result)
    self._send_auth_result(username, self.method, result)