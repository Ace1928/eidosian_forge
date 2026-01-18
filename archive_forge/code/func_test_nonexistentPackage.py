from __future__ import annotations
import os
from io import StringIO
from typing import TypeVar
from twisted.scripts import trial
from twisted.trial import runner
from twisted.trial.test import packages
def test_nonexistentPackage(self) -> str:
    d = self.runTrial('doesntexist')
    self.assertIn(d, 'doesntexist')
    self.assertIn(d, 'ModuleNotFound')
    self.assertIn(d, '[ERROR]')
    return d