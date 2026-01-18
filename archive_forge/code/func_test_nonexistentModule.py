from __future__ import annotations
import os
from io import StringIO
from typing import TypeVar
from twisted.scripts import trial
from twisted.trial import runner
from twisted.trial.test import packages
def test_nonexistentModule(self) -> str:
    d = self.runTrial('twisted.doesntexist')
    self.assertIn(d, '[ERROR]')
    self.assertIn(d, 'twisted.doesntexist')
    return d