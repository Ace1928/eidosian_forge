import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def _base_init(self, readermode):
    """Partial initialization for the NNTP protocol.
        This instance method is extracted for supporting the test code.
        """
    self.debugging = 0
    self.welcome = self._getresp()
    self._caps = None
    self.getcapabilities()
    self.readermode_afterauth = False
    if readermode and 'READER' not in self._caps:
        self._setreadermode()
        if not self.readermode_afterauth:
            self._caps = None
            self.getcapabilities()
    self.tls_on = False
    self.authenticated = False