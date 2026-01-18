from __future__ import annotations
from typing import Callable
from typing_extensions import NoReturn, Protocol
from twisted.python import randbytes
from twisted.trial import unittest
def test_withoutGetrandbits(self) -> None:
    """
        Test C{insecureRandom} without C{random.getrandbits}.
        """
    factory = randbytes.RandomFactory()
    factory.getrandbits = None
    self._check(factory.insecureRandom)