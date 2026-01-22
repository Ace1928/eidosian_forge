import random
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred.credentials import (
from twisted.cred.portal import Portal
from twisted.internet import defer, protocol
from twisted.persisted import styles
from twisted.python import failure, log, reflect
from twisted.python.compat import cmp, comparable
from twisted.python.components import registerAdapter
from twisted.spread import banana
from twisted.spread.flavors import (
from twisted.spread.interfaces import IJellyable, IUnjellyable
from twisted.spread.jelly import _newInstance, globalSecurity, jelly, unjelly
def _sendError(self, fail, requestID):
    """
        (internal) Send an error for a previously sent message.

        @param fail: The failure.
        @param requestID: The request ID.
        """
    if isinstance(fail, failure.Failure):
        if isinstance(fail.value, Jellyable) or self.security.isClassAllowed(fail.value.__class__):
            fail = fail.value
        elif not isinstance(fail, CopyableFailure):
            fail = failure2Copyable(fail, self.factory.unsafeTracebacks)
    if isinstance(fail, CopyableFailure):
        fail.unsafeTracebacks = self.factory.unsafeTracebacks
    self.sendCall(b'error', requestID, self.serialize(fail))