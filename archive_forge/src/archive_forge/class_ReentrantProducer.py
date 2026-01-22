import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
class ReentrantProducer(DummyProducer):
    """
    Similar to L{DummyProducer}, but with a resumeProducing method which calls
    back into an L{IConsumer} method of the consumer against which it is
    registered.

    @ivar consumer: The consumer with which this producer has been or will
    be registered.

    @ivar methodName: The name of the method to call on the consumer inside
    C{resumeProducing}.

    @ivar methodArgs: The arguments to pass to the consumer method invoked in
    C{resumeProducing}.
    """

    def __init__(self, consumer, methodName, *methodArgs):
        super().__init__()
        self.consumer = consumer
        self.methodName = methodName
        self.methodArgs = methodArgs

    def resumeProducing(self):
        super().resumeProducing()
        getattr(self.consumer, self.methodName)(*self.methodArgs)