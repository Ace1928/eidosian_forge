from zope.interface.verify import verifyClass
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IPushProducer
from twisted.trial.unittest import SynchronousTestCase
def writeSomeData(self, data):
    """
        Copy at most C{self._freeSpace} bytes from C{data} into C{self._written}.

        @return: A C{int} indicating how many bytes were copied from C{data}.
        """
    acceptLength = min(self._freeSpace, len(data))
    if acceptLength:
        self._freeSpace -= acceptLength
        self._written.append(data[:acceptLength])
    return acceptLength