from __future__ import (absolute_import, division, print_function)
import os
import socket
import random
import time
import uuid
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins.callback import CallbackBase
class PlainTextSocketAppender(object):

    def __init__(self, display, LE_API='data.logentries.com', LE_PORT=80, LE_TLS_PORT=443):
        self.LE_API = LE_API
        self.LE_PORT = LE_PORT
        self.LE_TLS_PORT = LE_TLS_PORT
        self.MIN_DELAY = 0.1
        self.MAX_DELAY = 10
        self.INVALID_TOKEN = '\n\nIt appears the LOGENTRIES_TOKEN parameter you entered is incorrect!\n\n'
        self.LINE_SEP = u'\u2028'
        self._display = display
        self._conn = None

    def open_connection(self):
        self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._conn.connect((self.LE_API, self.LE_PORT))

    def reopen_connection(self):
        self.close_connection()
        root_delay = self.MIN_DELAY
        while True:
            try:
                self.open_connection()
                return
            except Exception as e:
                self._display.vvvv(u'Unable to connect to Logentries: %s' % to_text(e))
            root_delay *= 2
            if root_delay > self.MAX_DELAY:
                root_delay = self.MAX_DELAY
            wait_for = root_delay + random.uniform(0, root_delay)
            try:
                self._display.vvvv('sleeping %s before retry' % wait_for)
                time.sleep(wait_for)
            except KeyboardInterrupt:
                raise

    def close_connection(self):
        if self._conn is not None:
            self._conn.close()

    def put(self, data):
        data = to_text(data, errors='surrogate_or_strict')
        multiline = data.replace(u'\n', self.LINE_SEP)
        multiline += u'\n'
        while True:
            try:
                self._conn.send(to_bytes(multiline, errors='surrogate_or_strict'))
            except socket.error:
                self.reopen_connection()
                continue
            break
        self.close_connection()