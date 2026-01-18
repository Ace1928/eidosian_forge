from __future__ import annotations
from typing_extensions import NoReturn
from twisted.python.monkey import MonkeyPatcher
from twisted.trial import unittest
def test_runWithPatchesRestoresOnException(self) -> None:
    """
        Test runWithPatches restores the original values even when the function
        raises an exception.
        """

    def _() -> NoReturn:
        self.assertEqual(self.testObject.foo, 'haha')
        self.assertEqual(self.testObject.bar, 'blahblah')
        raise RuntimeError('Something went wrong!')
    self.monkeyPatcher.addPatch(self.testObject, 'foo', 'haha')
    self.monkeyPatcher.addPatch(self.testObject, 'bar', 'blahblah')
    self.assertRaises(RuntimeError, self.monkeyPatcher.runWithPatches, _)
    self.assertEqual(self.testObject.foo, self.originalObject.foo)
    self.assertEqual(self.testObject.bar, self.originalObject.bar)