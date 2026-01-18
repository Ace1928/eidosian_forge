import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_redeclare_intrinsic(self):
    module = self.module()
    powi = module.declare_intrinsic('llvm.powi', [dbl])
    powi2 = module.declare_intrinsic('llvm.powi', [dbl])
    self.assertIs(powi, powi2)