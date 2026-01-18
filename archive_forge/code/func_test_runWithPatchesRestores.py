from __future__ import annotations
from typing_extensions import NoReturn
from twisted.python.monkey import MonkeyPatcher
from twisted.trial import unittest
def test_runWithPatchesRestores(self) -> None:
    """
        C{runWithPatches} should restore the original values after the function
        has executed.
        """
    self.monkeyPatcher.addPatch(self.testObject, 'foo', 'haha')
    self.assertEqual(self.originalObject.foo, self.testObject.foo)
    self.monkeyPatcher.runWithPatches(lambda: None)
    self.assertEqual(self.originalObject.foo, self.testObject.foo)