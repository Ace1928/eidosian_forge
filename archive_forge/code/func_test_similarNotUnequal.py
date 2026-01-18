from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_similarNotUnequal(self) -> None:
    """
        Structurally similar L{URL}s are not unequal (C{!=}) to each other.
        """
    u1 = URL.fromText('http://localhost/')
    u2 = URL.fromText('http://localhost/')
    self.assertFalse(u1 != u2, f'{u1!r} == {u2!r}')