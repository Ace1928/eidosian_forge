import os.path
import unittest
from Cython.TestUtils import TransformTest
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.ParseTreeTransforms import _calculate_pickle_checksums
from Cython.Compiler.Nodes import *
from Cython.Compiler import Main, Symtab, Options
class TestNormalizeTree(TransformTest):

    def test_parserbehaviour_is_what_we_coded_for(self):
        t = self.fragment(u'if x: y').root
        self.assertLines(u'\n(root): StatListNode\n  stats[0]: IfStatNode\n    if_clauses[0]: IfClauseNode\n      condition: NameNode\n      body: ExprStatNode\n        expr: NameNode\n', self.treetypes(t))

    def test_wrap_singlestat(self):
        t = self.run_pipeline([NormalizeTree(None)], u'if x: y')
        self.assertLines(u'\n(root): StatListNode\n  stats[0]: IfStatNode\n    if_clauses[0]: IfClauseNode\n      condition: NameNode\n      body: StatListNode\n        stats[0]: ExprStatNode\n          expr: NameNode\n', self.treetypes(t))

    def test_wrap_multistat(self):
        t = self.run_pipeline([NormalizeTree(None)], u'\n            if z:\n                x\n                y\n        ')
        self.assertLines(u'\n(root): StatListNode\n  stats[0]: IfStatNode\n    if_clauses[0]: IfClauseNode\n      condition: NameNode\n      body: StatListNode\n        stats[0]: ExprStatNode\n          expr: NameNode\n        stats[1]: ExprStatNode\n          expr: NameNode\n', self.treetypes(t))

    def test_statinexpr(self):
        t = self.run_pipeline([NormalizeTree(None)], u'\n            a, b = x, y\n        ')
        self.assertLines(u'\n(root): StatListNode\n  stats[0]: SingleAssignmentNode\n    lhs: TupleNode\n      args[0]: NameNode\n      args[1]: NameNode\n    rhs: TupleNode\n      args[0]: NameNode\n      args[1]: NameNode\n', self.treetypes(t))

    def test_wrap_offagain(self):
        t = self.run_pipeline([NormalizeTree(None)], u'\n            x\n            y\n            if z:\n                x\n        ')
        self.assertLines(u'\n(root): StatListNode\n  stats[0]: ExprStatNode\n    expr: NameNode\n  stats[1]: ExprStatNode\n    expr: NameNode\n  stats[2]: IfStatNode\n    if_clauses[0]: IfClauseNode\n      condition: NameNode\n      body: StatListNode\n        stats[0]: ExprStatNode\n          expr: NameNode\n', self.treetypes(t))

    def test_pass_eliminated(self):
        t = self.run_pipeline([NormalizeTree(None)], u'pass')
        self.assertTrue(len(t.stats) == 0)