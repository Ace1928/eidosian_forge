from __future__ import (absolute_import, division, print_function)
import os
import socket
import random
import time
import uuid
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins.callback import CallbackBase
class TLSSocketAppender(PlainTextSocketAppender):

    def open_connection(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH, cafile=certifi.where())
        sock = context.wrap_socket(sock=sock, do_handshake_on_connect=True, suppress_ragged_eofs=True)
        sock.connect((self.LE_API, self.LE_TLS_PORT))
        self._conn = sock