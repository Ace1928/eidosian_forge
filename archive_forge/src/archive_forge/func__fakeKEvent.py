from __future__ import annotations
import errno
from zope.interface import implementer
from twisted.trial.unittest import TestCase
def _fakeKEvent(*args: object, **kwargs: object) -> None:
    """
    Do nothing.
    """