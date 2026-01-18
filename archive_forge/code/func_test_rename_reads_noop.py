from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import unittest
from pasta.augment import rename
from pasta.base import scope
from pasta.base import test_utils
def test_rename_reads_noop(self):
    src = 'aaa.bbb.ccc()'
    t = ast.parse(src)
    sc = scope.analyze(t)
    rename._rename_reads(sc, t, 'aaa.bbb.ccc.ddd', 'xxx.yyy')
    rename._rename_reads(sc, t, 'bbb.aaa', 'xxx.yyy')
    self.checkAstsEqual(t, ast.parse(src))