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
class CopyableFailure(failure.Failure, Copyable):
    """
    A L{flavors.RemoteCopy} and L{flavors.Copyable} version of
    L{twisted.python.failure.Failure} for serialization.
    """
    unsafeTracebacks = 0

    def getStateToCopy(self):
        """
        Collect state related to the exception which occurred, discarding
        state which cannot reasonably be serialized.
        """
        state = self.__dict__.copy()
        state['tb'] = None
        state['frames'] = []
        state['stack'] = []
        state['value'] = str(self.value)
        if isinstance(self.type, bytes):
            state['type'] = self.type
        else:
            state['type'] = reflect.qual(self.type).encode('utf-8')
        if self.unsafeTracebacks:
            state['traceback'] = self.getTraceback()
        else:
            state['traceback'] = 'Traceback unavailable\n'
        return state