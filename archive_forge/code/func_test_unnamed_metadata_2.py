import copy
import itertools
import pickle
import re
import textwrap
import unittest
from . import TestCase
from llvmlite import ir
from llvmlite import binding as llvm
def test_unnamed_metadata_2(self):
    mod = self.module()
    m0 = mod.add_metadata([int32(123), 'kernel'])
    m1 = mod.add_metadata([int64(456), m0])
    m2 = mod.add_metadata([int64(456), m0])
    self.assertIs(m2, m1)
    mod.add_metadata([m0, m1, m2])
    self.assert_ir_line('!0 = !{ i32 123, !"kernel" }', mod)
    self.assert_ir_line('!1 = !{ i64 456, !0 }', mod)
    self.assert_ir_line('!2 = !{ !0, !1, !1 }', mod)