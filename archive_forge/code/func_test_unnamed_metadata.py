import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_unnamed_metadata(self):
    mod = self.module()
    mod.add_metadata([int32(123), int8(42)])
    self.assert_ir_line('!0 = !{ i32 123, i8 42 }', mod)
    self.assert_valid_ir(mod)