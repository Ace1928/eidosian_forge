from twisted.protocols import pcp
from twisted.trial import unittest
class ConsumerInterfaceTest:
    """Test ProducerConsumerProxy as a Consumer.

    Normally we have ProducingServer -> ConsumingTransport.

    If I am to go between (Server -> Shaper -> Transport), I have to
    play the role of Consumer convincingly for the ProducingServer.
    """

    def setUp(self):
        self.underlying = DummyConsumer()
        self.consumer = self.proxyClass(self.underlying)
        self.producer = DummyProducer(self.consumer)

    def testRegisterPush(self):
        self.consumer.registerProducer(self.producer, True)
        self.assertFalse(self.producer.resumed)

    def testUnregister(self):
        self.consumer.registerProducer(self.producer, False)
        self.consumer.unregisterProducer()
        self.producer.resumed = False
        self.consumer.resumeProducing()
        self.assertFalse(self.producer.resumed)

    def testFinish(self):
        self.consumer.registerProducer(self.producer, False)
        self.consumer.finish()
        self.producer.resumed = False
        self.consumer.resumeProducing()
        self.assertFalse(self.producer.resumed)