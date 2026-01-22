from breezy import transport
from breezy.tests import TestCaseWithMemoryTransport
from breezy.trace import mutter
from breezy.transport.log import TransportLogDecorator
class DummyReadvTransport:
    base = 'dummy:///'

    def readv(self, filename, offset_length_pairs):
        yield (0, 'abcdefghij')

    def abspath(self, path):
        return self.base + path