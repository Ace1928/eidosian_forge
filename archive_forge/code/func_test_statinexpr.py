import os.path
import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.ParseTreeTransforms import _calculate_pickle_checksums
from Cython.Compiler.Nodes import *
from Cython.Compiler import Main, Symtab, Options
def test_statinexpr(self):
    t = self.run_pipeline([NormalizeTree(None)], u'\n            a, b = x, y\n        ')
    self.assertLines(u'\n(root): StatListNode\n  stats[0]: SingleAssignmentNode\n    lhs: TupleNode\n      args[0]: NameNode\n      args[1]: NameNode\n    rhs: TupleNode\n      args[0]: NameNode\n      args[1]: NameNode\n', self.treetypes(t))