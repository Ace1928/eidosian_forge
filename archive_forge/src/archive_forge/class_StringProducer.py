from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import (
@implementer(IBodyProducer)
class StringProducer:
    """
    L{StringProducer} is a dummy body producer.

    @ivar stopped: A flag which indicates whether or not C{stopProducing} has
        been called.
    @ivar consumer: After C{startProducing} is called, the value of the
        C{consumer} argument to that method.
    @ivar finished: After C{startProducing} is called, a L{Deferred} which was
        returned by that method.  L{StringProducer} will never fire this
        L{Deferred}.
    """
    stopped = False

    def __init__(self, length):
        self.length = length

    def startProducing(self, consumer):
        self.consumer = consumer
        self.finished = Deferred()
        return self.finished

    def stopProducing(self):
        self.stopped = True

    def pauseProducing(self):
        pass

    def resumeProducing(self):
        pass