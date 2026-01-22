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
def _ebBadAuth(self, reason):
    """
        The final errback in the authentication chain.  If the reason is
        error.IgnoreAuthentication, we simply return; the authentication
        method has sent its own response.  Otherwise, send a failure message
        and (if the method is not 'none') increment the number of login
        attempts.

        @type reason: L{twisted.python.failure.Failure}
        """
    if reason.check(error.IgnoreAuthentication):
        return
    if self.method != b'none':
        self._log.debug('{user!r} failed auth {method!r}', user=self.user, method=self.method)
        if reason.check(UnauthorizedLogin):
            self._log.debug('unauthorized login: {message}', message=reason.getErrorMessage())
        elif reason.check(error.ConchError):
            self._log.debug('reason: {reason}', reason=reason.getErrorMessage())
        else:
            self._log.failure('Error checking auth for user {user}', failure=reason, user=self.user)
        self.loginAttempts += 1
        if self.loginAttempts > self.attemptsBeforeDisconnect:
            self.transport.sendDisconnect(transport.DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE, b'too many bad auths')
            return
    self.transport.sendPacket(MSG_USERAUTH_FAILURE, NS(b','.join(self.supportedAuthentications)) + b'\x00')