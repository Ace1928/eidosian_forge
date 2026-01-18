from __future__ import annotations
from typing_extensions import NoReturn
from twisted.python.monkey import MonkeyPatcher
from twisted.trial import unittest
def test_runWithPatchesDecoration(self) -> None:
    """
        runWithPatches should run the given callable, passing in all arguments
        and keyword arguments, and return the return value of the callable.
        """
    log: list[tuple[int, int, int | None]] = []

    def f(a: int, b: int, c: int | None=None) -> str:
        log.append((a, b, c))
        return 'foo'
    result = self.monkeyPatcher.runWithPatches(f, 1, 2, c=10)
    self.assertEqual('foo', result)
    self.assertEqual([(1, 2, 10)], log)