from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_positional_only_argument_annotations(self):
    self.flakes('\n        from x import C\n\n        def f(c: C, /): ...\n        ')