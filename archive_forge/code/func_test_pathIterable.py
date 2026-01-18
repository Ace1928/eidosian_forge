from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def test_pathIterable(self) -> None:
    """
        When a L{URL} is created with a C{path} argument, the C{path} is
        converted into a tuple.
        """
    url = URL(path=['hello', 'world'])
    self.assertEqual(url.path, ('hello', 'world'))