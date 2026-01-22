from twisted.protocols import pcp
from twisted.trial import unittest
class PCPII_PullProducerTests(PullProducerTest, unittest.TestCase):

    class proxyClass(pcp.ProducerConsumerProxy):
        iAmStreaming = False