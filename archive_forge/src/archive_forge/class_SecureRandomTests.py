from __future__ import annotations
from typing import Callable
from typing_extensions import NoReturn, Protocol
from twisted.python import randbytes
from twisted.trial import unittest
class SecureRandomTests(SecureRandomTestCaseBase, unittest.TestCase):
    """
    Test secureRandom under normal conditions.
    """

    def test_normal(self) -> None:
        """
        L{randbytes.secureRandom} should return a string of the requested
        length and make some effort to make its result otherwise unpredictable.
        """
        self._check(randbytes.secureRandom)