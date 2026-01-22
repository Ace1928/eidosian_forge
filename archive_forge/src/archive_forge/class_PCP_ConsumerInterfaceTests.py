from twisted.protocols import pcp
from twisted.trial import unittest
class PCP_ConsumerInterfaceTests(ConsumerInterfaceTest, unittest.TestCase):
    proxyClass = pcp.BasicProducerConsumerProxy