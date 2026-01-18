import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
def timeoutFunc(self):
    """
        This method is called when the timeout is triggered.

        By default it calls I{loseConnection}.  Override this if you want
        something else to happen.
        """
    self.loseConnection()