from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReactorWin32Events
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.failure import Failure
from twisted.python.threadable import getThreadID, isInIOThread

        Event handlers added with L{IReactorWin32Events.addEvent} do not have
        C{connectionLost} called on them if they are still active when the
        reactor shuts down.
        