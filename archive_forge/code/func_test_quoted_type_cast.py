from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_quoted_type_cast(self):
    self.flakes("\n        from typing import cast, Optional\n\n        maybe_int = cast('Optional[int]', 42)\n        ")