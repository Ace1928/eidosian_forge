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
def _parse_userauth_info_response(self, m):
    if not self.transport.server_mode:
        raise SSHException('Illegal info response from server')
    n = m.get_int()
    responses = []
    for i in range(n):
        responses.append(m.get_text())
    result = self.transport.server_object.check_auth_interactive_response(responses)
    if isinstance(result, InteractiveQuery):
        self._interactive_query(result)
        return
    self._send_auth_result(self.auth_username, 'keyboard-interactive', result)