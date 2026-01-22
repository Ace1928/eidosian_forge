from zope.interface.verify import verifyClass
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IPushProducer
from twisted.trial.unittest import SynchronousTestCase
class MemoryFile(FileDescriptor):
    """
    A L{FileDescriptor} customization which writes to a Python list in memory
    with certain limitations.

    @ivar _written: A C{list} of C{bytes} which have been accepted as written.

    @ivar _freeSpace: A C{int} giving the number of bytes which will be accepted
        by future writes.
    """
    connected = True

    def __init__(self):
        FileDescriptor.__init__(self, reactor=object())
        self._written = []
        self._freeSpace = 0

    def startWriting(self):
        pass

    def stopWriting(self):
        pass

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