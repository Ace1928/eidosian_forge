from __future__ import (absolute_import, division, print_function)
import os
import socket
import random
import time
import uuid
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins.callback import CallbackBase
def _initialize_connections(self):
    if not self.disabled:
        if self.use_tls:
            self._display.vvvv('Connecting to %s:%s with TLS' % (self.api_url, self.api_tls_port))
            self._appender = TLSSocketAppender(display=self._display, LE_API=self.api_url, LE_TLS_PORT=self.api_tls_port)
        else:
            self._display.vvvv('Connecting to %s:%s' % (self.api_url, self.api_port))
            self._appender = PlainTextSocketAppender(display=self._display, LE_API=self.api_url, LE_PORT=self.api_port)
        self._appender.reopen_connection()