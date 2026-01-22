from twisted.protocols import pcp
from twisted.trial import unittest
class PCPII_ConsumerInterfaceTests(ConsumerInterfaceTest, unittest.TestCase):
    proxyClass = pcp.ProducerConsumerProxy