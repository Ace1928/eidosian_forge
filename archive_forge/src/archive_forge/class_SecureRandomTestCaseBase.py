from __future__ import annotations
from typing import Callable
from typing_extensions import NoReturn, Protocol
from twisted.python import randbytes
from twisted.trial import unittest
class SecureRandomTestCaseBase:
    """
    Base class for secureRandom test cases.
    """

    def _check(self: _SupportsAssertions, source: Callable[[int], bytes]) -> None:
        """
        The given random bytes source should return the number of bytes
        requested each time it is called and should probably not return the
        same bytes on two consecutive calls (although this is a perfectly
        legitimate occurrence and rejecting it may generate a spurious failure
        -- maybe we'll get lucky and the heat death with come first).
        """
        for nbytes in range(17, 25):
            s = source(nbytes)
            self.assertEqual(len(s), nbytes)
            s2 = source(nbytes)
            self.assertEqual(len(s2), nbytes)
            self.assertNotEqual(s2, s)