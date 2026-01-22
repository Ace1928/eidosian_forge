import unittest
import sys
import fixtures  # type: ignore
import typing
from autopage import command
def _setUp(self) -> None:
    self.addCleanup(setattr, sys, 'platform', sys.platform)
    sys.platform = self.platform