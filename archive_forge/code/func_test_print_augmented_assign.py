from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_print_augmented_assign(self):
    self.flakes('print += 1')