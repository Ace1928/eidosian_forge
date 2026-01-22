from typing import Type
from twisted.internet import error
from twisted.internet.protocol import Protocol, connectionDone
from twisted.persisted import styles
from twisted.python.failure import Failure
from twisted.python.reflect import prefixedMethods
from twisted.words.im.locals import OFFLINE, OfflineError
def _loginFailed(self, reason):
    """Errorback for L{logOn}.

        @type reason: Failure

        @returns: I{reason}, for further processing in the callback chain.
        @returntype: Failure
        """
    self._isConnecting = 0
    self._isOnline = 0
    return reason