from __future__ import annotations
from typing import Callable
from typing_extensions import NoReturn, Protocol
from twisted.python import randbytes
from twisted.trial import unittest
def test_withoutAnything(self) -> None:
    """
        Remove all secure sources and assert it raises a failure. Then try the
        fallback parameter.
        """
    self.factory._osUrandom = self.errorFactory
    self.assertRaises(randbytes.SecureRandomNotAvailable, self.factory.secureRandom, 18)

    def wrapper() -> bytes:
        return self.factory.secureRandom(18, fallback=True)
    s = self.assertWarns(RuntimeWarning, 'urandom unavailable - proceeding with non-cryptographically secure random source', __file__, wrapper)
    self.assertEqual(len(s), 18)