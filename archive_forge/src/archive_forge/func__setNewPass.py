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
def _setNewPass(self, np):
    """
        Called back when we are choosing a new password.  Get the old password
        and send the authentication message with both.

        @param np: the new password as entered by the user
        @type np: L{bytes}
        """
    op = self._oldPass
    self._oldPass = None
    self.askForAuth(b'password', b'\xff' + NS(op) + NS(np))