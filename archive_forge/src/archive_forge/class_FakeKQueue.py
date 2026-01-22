from __future__ import annotations
import errno
from zope.interface import implementer
from twisted.trial.unittest import TestCase
class FakeKQueue:
    """
            A fake KQueue that raises L{errno.EINTR} when C{control} is called,
            like a real KQueue would if it was interrupted.
            """

    def control(self, *args: object, **kwargs: object) -> None:
        raise OSError(errno.EINTR, 'Interrupted')