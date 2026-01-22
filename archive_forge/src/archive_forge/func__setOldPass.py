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
def _setOldPass(self, op):
    """
        Called back when we are choosing a new password.  Simply store the old
        password for now.

        @param op: the old password as entered by the user
        @type op: L{bytes}
        """
    self._oldPass = op