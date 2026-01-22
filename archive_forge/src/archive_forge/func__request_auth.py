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
def _request_auth(self):
    m = Message()
    m.add_byte(cMSG_SERVICE_REQUEST)
    m.add_string('ssh-userauth')
    self.transport._send_message(m)