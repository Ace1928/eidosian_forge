from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def test_pushProducer(self):
    """
        Test a push producer registered against a loopback transport.
        """

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
    d = self._producertest(PushProducer)

    def finished(results):
        client, server = results
        self.assertFalse(server.producer.resumed, 'Streaming producer should not have been resumed.')
    d.addCallback(finished)
    return d