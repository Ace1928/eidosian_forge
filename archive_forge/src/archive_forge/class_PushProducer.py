from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
@implementer(IPushProducer)
class PushProducer:
    resumed = False

    def __init__(self, toProduce):
        self.toProduce = toProduce

    def resumeProducing(self):
        self.resumed = True

    def start(self, consumer):
        self.consumer = consumer
        consumer.registerProducer(self, True)
        self._produceAndSchedule()

    def _produceAndSchedule(self):
        if self.toProduce:
            self.consumer.write(self.toProduce.pop(0))
            reactor.callLater(0, self._produceAndSchedule)
        else:
            self.consumer.unregisterProducer()