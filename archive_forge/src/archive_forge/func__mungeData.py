import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
def _mungeData(self, data):
    if self.lengthLimit and len(data) > self.lengthLimit:
        data = data[:self.lengthLimit - 12] + '<... elided>'
    return data