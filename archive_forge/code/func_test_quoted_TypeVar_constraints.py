from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_quoted_TypeVar_constraints(self):
    self.flakes("\n        from typing import TypeVar, Optional\n\n        T = TypeVar('T', 'str', 'Optional[int]', bytes)\n        ")