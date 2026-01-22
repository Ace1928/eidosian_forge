import os
import socket
import socketserver
import sys
import time
import paramiko
from .. import osutils, trace, urlutils
from ..transport import ssh
from . import test_server
class SFTPFullAbsoluteServer(SFTPServer):
    """A test server for sftp transports, using absolute urls and ssh."""

    def get_url(self):
        """See breezy.transport.Server.get_url."""
        homedir = self._homedir
        if sys.platform != 'win32':
            homedir = homedir[1:]
        return self._get_sftp_url(urlutils.escape(homedir))