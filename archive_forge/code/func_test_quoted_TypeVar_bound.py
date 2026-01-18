from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_quoted_TypeVar_bound(self):
    self.flakes("\n        from typing import TypeVar, Optional, List\n\n        T = TypeVar('T', bound='Optional[int]')\n        S = TypeVar('S', int, bound='List[int]')\n        ")