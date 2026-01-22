import struct
from twisted.conch import error, interfaces
from twisted.conch.ssh import keys, service, transport
from twisted.conch.ssh.common import NS, getNS
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, reactor
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString
def _cbPassword(self, password):
    """
        Called back when the user gives a password.  Send the request to the
        server.

        @param password: the password the user entered
        @type password: L{bytes}
        """
    self.askForAuth(b'password', b'\x00' + NS(password))