from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_quoted_type_cast_renamed_import(self):
    self.flakes("\n        from typing import cast as tsac, Optional as Maybe\n\n        maybe_int = tsac('Maybe[int]', 42)\n        ")