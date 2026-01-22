from __future__ import annotations
from typing import Callable
from typing_extensions import NoReturn, Protocol
from twisted.python import randbytes
from twisted.trial import unittest
class RandomBaseTests(SecureRandomTestCaseBase, unittest.SynchronousTestCase):
    """
    'Normal' random test cases.
    """

    def test_normal(self) -> None:
        """
        Test basic case.
        """
        self._check(randbytes.insecureRandom)

    def test_withoutGetrandbits(self) -> None:
        """
        Test C{insecureRandom} without C{random.getrandbits}.
        """
        factory = randbytes.RandomFactory()
        factory.getrandbits = None
        self._check(factory.insecureRandom)