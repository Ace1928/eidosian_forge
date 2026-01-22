from __future__ import annotations
from typing import Iterable
from typing_extensions import Protocol
from twisted.trial.unittest import SynchronousTestCase
from ..url import URL
def assertRaised(raised: _HasException, expectation: str, name: str) -> None:
    self.assertEqual(str(raised.exception), 'expected {} for {}, got {}'.format(expectation, name, '<unexpected>'))