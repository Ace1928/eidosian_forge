import textwrap
from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.test_other import Test as TestOther
from pyflakes.test.test_imports import Test as TestImports
from pyflakes.test.test_undefined_names import Test as TestUndefinedNames
from pyflakes.test.harness import TestCase, skip
def test_singleUnderscoreInDoctest(self):
    self.flakes('\n        def func():\n            """A docstring\n\n            >>> func()\n            1\n            >>> _\n            1\n            """\n            return 1\n        ')