from __future__ import annotations
import os
from io import StringIO
from typing import TypeVar
from twisted.scripts import trial
from twisted.trial import runner
from twisted.trial.test import packages
def test_regularRun(self) -> str:
    d = self.runTrial('package.test_module')
    self.assertNotIn(d, '[ERROR]')
    self.assertNotIn(d, 'IOError')
    self.assertIn(d, 'OK')
    self.assertIn(d, 'PASSED (successes=1)')
    return d