from unittest import skipIf
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_writeUnicode(self) -> None:
    """
        L{_pollingfile._PollableWritePipe.write} raises a C{TypeError} if an
        attempt is made to append unicode data to the output buffer.
        """
    p = _pollingfile._PollableWritePipe(1, lambda: None)
    self.assertRaises(TypeError, p.write, 'test')