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
def _ebMaybeBadAuth(self, reason):
    """
        An intermediate errback.  If the reason is
        error.NotEnoughAuthentication, we send a MSG_USERAUTH_FAILURE, but
        with the partial success indicator set.

        @type reason: L{twisted.python.failure.Failure}
        """
    reason.trap(error.NotEnoughAuthentication)
    self.transport.sendPacket(MSG_USERAUTH_FAILURE, NS(b','.join(self.supportedAuthentications)) + b'\xff')