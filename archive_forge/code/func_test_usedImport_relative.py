from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_usedImport_relative(self):
    self.flakes('from . import fu; assert fu')
    self.flakes('from .bar import fu; assert fu')
    self.flakes('from .. import fu; assert fu')
    self.flakes('from ..bar import fu as baz; assert baz')