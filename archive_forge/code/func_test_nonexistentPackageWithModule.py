from __future__ import annotations
import os
from io import StringIO
from typing import TypeVar
from twisted.scripts import trial
from twisted.trial import runner
from twisted.trial.test import packages
def test_nonexistentPackageWithModule(self) -> str:
    d = self.runTrial('doesntexist.barney')
    self.assertIn(d, 'doesntexist.barney')
    self.assertIn(d, 'ObjectNotFound')
    self.assertIn(d, '[ERROR]')
    return d