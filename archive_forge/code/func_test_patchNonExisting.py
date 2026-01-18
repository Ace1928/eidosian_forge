from __future__ import annotations
from typing_extensions import NoReturn
from twisted.python.monkey import MonkeyPatcher
from twisted.trial import unittest
def test_patchNonExisting(self) -> None:
    """
        Patching a non-existing attribute fails with an C{AttributeError}.
        """
    self.monkeyPatcher.addPatch(self.testObject, 'nowhere', 'blow up please')
    self.assertRaises(AttributeError, self.monkeyPatcher.patch)