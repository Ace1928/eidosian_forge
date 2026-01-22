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
def _reentrantStreamingProducerTest(self, methodName):
    descriptor = SillyDescriptor()
    if methodName == 'writeSequence':
        data = [b's', b'p', b'am']
    else:
        data = b'spam'
    producer = ReentrantProducer(descriptor, methodName, data)
    descriptor.registerProducer(producer, streaming=True)
    getattr(descriptor, methodName)(data)
    self.assertEqual(producer.events, ['pause'])
    del producer.events[:]
    descriptor.doWrite()
    self.assertEqual(producer.events, ['resume', 'pause'])
    del producer.events[:]
    descriptor.doWrite()
    self.assertEqual(producer.events, ['resume', 'pause'])