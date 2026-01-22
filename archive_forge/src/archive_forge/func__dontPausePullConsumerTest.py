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
def _dontPausePullConsumerTest(self, methodName):
    """
        Pull consumers don't get their C{pauseProducing} method called if the
        descriptor buffer fills up.

        @param _methodName: Either 'write', or 'writeSequence', indicating
            which transport method to write data to.
        """
    descriptor = SillyDescriptor()
    producer = DummyProducer()
    descriptor.registerProducer(producer, streaming=False)
    self.assertEqual(producer.events, ['resume'])
    del producer.events[:]
    if methodName == 'writeSequence':
        descriptor.writeSequence([b'1', b'2', b'3', b'4'])
    else:
        descriptor.write(b'1234')
    self.assertEqual(producer.events, [])