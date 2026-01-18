import sys
from zope.interface import implementer
from twisted.internet import protocol, stdio
from twisted.internet.interfaces import IHalfCloseableProtocol
from twisted.python import log, reflect

        This may only be invoked after C{readConnectionLost}.  If it happens
        otherwise, mark it as an error and shut down.
        