from twisted.protocols import pcp
from twisted.trial import unittest
class proxyClass(pcp.ProducerConsumerProxy):
    iAmStreaming = False

    def _writeSomeData(self, data):
        pcp.ProducerConsumerProxy._writeSomeData(self, data[:100])
        return min(len(data), 100)