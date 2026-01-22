import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
class ConchTestRealm:
    """
    A realm which expects a particular avatarId to log in once and creates a
    L{ConchTestAvatar} for that request.

    @ivar expectedAvatarID: The only avatarID that this realm will produce an
        avatar for.

    @ivar avatar: A reference to the avatar after it is requested.
    """
    avatar = None

    def __init__(self, expectedAvatarID):
        self.expectedAvatarID = expectedAvatarID

    def requestAvatar(self, avatarID, mind, *interfaces):
        """
        Return a new L{ConchTestAvatar} if the avatarID matches the expected one
        and this is the first avatar request.
        """
        if avatarID == self.expectedAvatarID:
            if self.avatar is not None:
                raise UnauthorizedLogin('Only one login allowed')
            self.avatar = ConchTestAvatar()
            return (interfaces[0], self.avatar, self.avatar.logout)
        raise UnauthorizedLogin(f'Only {self.expectedAvatarID!r} may log in, not {avatarID!r}')