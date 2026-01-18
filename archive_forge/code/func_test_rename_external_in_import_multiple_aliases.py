from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import unittest
from pasta.augment import rename
from pasta.base import scope
from pasta.base import test_utils
def test_rename_external_in_import_multiple_aliases(self):
    src = 'import aaa, aaa.bbb, aaa.bbb.ccc'
    t = ast.parse(src)
    self.assertTrue(rename.rename_external(t, 'aaa.bbb', 'xxx.yyy'))
    self.checkAstsEqual(t, ast.parse('import aaa, xxx.yyy, xxx.yyy.ccc'))