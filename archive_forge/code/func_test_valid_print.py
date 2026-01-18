from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_valid_print(self):
    self.flakes('\n        print("Hello")\n        ')