from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import unittest
from pasta.augment import rename
from pasta.base import scope
from pasta.base import test_utils
def test_rename_external_in_importfrom(self):
    src = 'from aaa.bbb.ccc import ddd\nddd.foo()'
    t = ast.parse(src)
    self.assertTrue(rename.rename_external(t, 'aaa.bbb', 'xxx.yyy'))
    self.checkAstsEqual(t, ast.parse('from xxx.yyy.ccc import ddd\nddd.foo()'))
    t = ast.parse(src)
    self.assertTrue(rename.rename_external(t, 'aaa.bbb.ccc', 'xxx.yyy'))
    self.checkAstsEqual(t, ast.parse('from xxx.yyy import ddd\nddd.foo()'))
    t = ast.parse(src)
    self.assertFalse(rename.rename_external(t, 'bbb', 'xxx.yyy'))
    self.checkAstsEqual(t, ast.parse(src))