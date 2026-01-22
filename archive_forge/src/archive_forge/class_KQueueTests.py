from __future__ import annotations
import errno
from zope.interface import implementer
from twisted.trial.unittest import TestCase
class KQueueTests(TestCase):
    """
    These are tests for L{KQueueReactor}'s implementation, not its real world
    behaviour. For that, look at
    L{twisted.internet.test.reactormixins.ReactorBuilder}.
    """
    skip = kqueueSkip

    def test_EINTR(self) -> None:
        """
        L{KQueueReactor} handles L{errno.EINTR} in C{doKEvent} by returning.
        """

        class FakeKQueue:
            """
            A fake KQueue that raises L{errno.EINTR} when C{control} is called,
            like a real KQueue would if it was interrupted.
            """

            def control(self, *args: object, **kwargs: object) -> None:
                raise OSError(errno.EINTR, 'Interrupted')
        reactor = KQueueReactor(makeFakeKQueue(FakeKQueue, _fakeKEvent))
        reactor.doKEvent(0)