from typing import Type
from twisted.internet import error
from twisted.internet.protocol import Protocol, connectionDone
from twisted.persisted import styles
from twisted.python.failure import Failure
from twisted.python.reflect import prefixedMethods
from twisted.words.im.locals import OFFLINE, OfflineError
class AbstractClientMixin:
    """Designed to be mixed in to a Protocol implementing class.

    Inherit from me first.

    @ivar _logonDeferred: Fired when I am done logging in.
    """
    _protoBase: Type[Protocol] = None

    def __init__(self, account, chatui, logonDeferred):
        for base in self.__class__.__bases__:
            if issubclass(base, Protocol):
                self.__class__._protoBase = base
                break
        else:
            pass
        self.account = account
        self.chat = chatui
        self._logonDeferred = logonDeferred

    def connectionMade(self):
        self._protoBase.connectionMade(self)

    def connectionLost(self, reason: Failure=connectionDone) -> None:
        self.account._clientLost(self, reason)
        self.unregisterAsAccountClient()
        return self._protoBase.connectionLost(self, reason)

    def unregisterAsAccountClient(self):
        """Tell the chat UI that I have `signed off'."""
        self.chat.unregisterAccountClient(self)