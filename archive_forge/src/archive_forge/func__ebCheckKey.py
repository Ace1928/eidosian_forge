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
def _ebCheckKey(self, reason, packet):
    """
        Called back if the user did not sent a signature.  If reason is
        error.ValidPublicKey then this key is valid for the user to
        authenticate with.  Send MSG_USERAUTH_PK_OK.
        """
    reason.trap(error.ValidPublicKey)
    self.transport.sendPacket(MSG_USERAUTH_PK_OK, packet)
    return failure.Failure(error.IgnoreAuthentication())