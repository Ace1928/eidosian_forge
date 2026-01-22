import os
from typing import Any, Dict, Optional, Tuple, Union
from zope.interface import Attribute, Interface, implementer
from twisted.cred import error
from twisted.cred.credentials import (
from twisted.internet import defer
from twisted.internet.defer import Deferred
from twisted.logger import Logger
from twisted.python import failure
@implementer(ICredentialsChecker)
class AllowAnonymousAccess:
    """
    A credentials checker that unconditionally grants anonymous access.

    @cvar credentialInterfaces: Tuple containing L{IAnonymous}.
    """
    credentialInterfaces = (IAnonymous,)

    def requestAvatarId(self, credentials):
        """
        Succeed with the L{ANONYMOUS} avatar ID.

        @return: L{Deferred} that fires with L{twisted.cred.checkers.ANONYMOUS}
        """
        return defer.succeed(ANONYMOUS)