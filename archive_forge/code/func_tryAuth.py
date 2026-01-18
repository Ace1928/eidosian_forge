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
def tryAuth(self, kind):
    """
        Dispatch to an authentication method.

        @param kind: the authentication method
        @type kind: L{bytes}
        """
    kind = nativeString(kind.replace(b'-', b'_'))
    self._log.debug('trying to auth with {kind}', kind=kind)
    f = getattr(self, 'auth_' + kind, None)
    if f:
        return f()