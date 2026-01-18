from __future__ import annotations
import os
from io import StringIO
from typing import TypeVar
from twisted.scripts import trial
from twisted.trial import runner
from twisted.trial.test import packages
def test_recurseImportErrors(self) -> str:
    d = self.runTrial('package2')
    self.assertIn(d, '[ERROR]')
    self.assertIn(d, 'package2')
    self.assertIn(d, 'test_module')
    self.assertIn(d, _noModuleError)
    self.assertNotIn(d, '<module ')
    self.assertNotIn(d, 'IOError')
    return d