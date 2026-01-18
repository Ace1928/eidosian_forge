from zope.interface.verify import verifyClass
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IPushProducer
from twisted.trial.unittest import SynchronousTestCase
def test_writeSequenceWithUnicodeRaisesException(self):
    """
        L{FileDescriptor.writeSequence} doesn't accept unicode data.
        """
    fileDescriptor = FileDescriptor(reactor=object())
    self.assertRaises(TypeError, fileDescriptor.writeSequence, [b'foo', 'bar', b'baz'])