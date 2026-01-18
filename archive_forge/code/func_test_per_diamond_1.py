import unittest
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.tests import TestCase
from . import refprune_proto as proto
def test_per_diamond_1(self):
    mod, stats = self.check(self.per_diamond_1)
    self.assertEqual(stats.diamond, 2)