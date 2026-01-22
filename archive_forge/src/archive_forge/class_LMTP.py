import socket
import io
import re
import email.utils
import email.message
import email.generator
import base64
import hmac
import copy
import datetime
import sys
from email.base64mime import body_encode as encode_base64
class LMTP(SMTP):
    """LMTP - Local Mail Transfer Protocol

    The LMTP protocol, which is very similar to ESMTP, is heavily based
    on the standard SMTP client. It's common to use Unix sockets for
    LMTP, so our connect() method must support that as well as a regular
    host:port server.  local_hostname and source_address have the same
    meaning as they do in the SMTP class.  To specify a Unix socket,
    you must use an absolute path as the host, starting with a '/'.

    Authentication is supported, using the regular SMTP mechanism. When
    using a Unix socket, LMTP generally don't support or require any
    authentication, but your mileage might vary."""
    ehlo_msg = 'lhlo'

    def __init__(self, host='', port=LMTP_PORT, local_hostname=None, source_address=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
        """Initialize a new instance."""
        super().__init__(host, port, local_hostname=local_hostname, source_address=source_address, timeout=timeout)

    def connect(self, host='localhost', port=0, source_address=None):
        """Connect to the LMTP daemon, on either a Unix or a TCP socket."""
        if host[0] != '/':
            return super().connect(host, port, source_address=source_address)
        if self.timeout is not None and (not self.timeout):
            raise ValueError('Non-blocking socket (timeout=0) is not supported')
        try:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            if self.timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                self.sock.settimeout(self.timeout)
            self.file = None
            self.sock.connect(host)
        except OSError:
            if self.debuglevel > 0:
                self._print_debug('connect fail:', host)
            if self.sock:
                self.sock.close()
            self.sock = None
            raise
        code, msg = self.getreply()
        if self.debuglevel > 0:
            self._print_debug('connect:', msg)
        return (code, msg)