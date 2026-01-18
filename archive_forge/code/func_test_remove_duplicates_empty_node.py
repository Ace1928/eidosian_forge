from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import traceback
import unittest
import pasta
from pasta.augment import import_utils
from pasta.base import ast_utils
from pasta.base import test_utils
from pasta.base import scope
import a
import b
from my_module import a
import b
from my_module import c
from my_module import a, b
from my_module import a as a_mod, b as unused_b_mod
import c as c_mod, d as unused_d_mod
import a
import b
import c
import b
import d
import a, b
import b, c
import d, a, e, f
import a, b, c
import b, c
import a.b
from a import b
import a
import a as ax
import a as ax2
import a as ax
def test_remove_duplicates_empty_node(self):
    src = '\nimport a, b, c\nimport b, c\n'
    tree = ast.parse(src)
    self.assertTrue(import_utils.remove_duplicates(tree))
    self.assertEqual(len(tree.body), 1)
    self.assertEqual(len(tree.body[0].names), 3)
    self.assertEqual(tree.body[0].names[0].name, 'a')
    self.assertEqual(tree.body[0].names[1].name, 'b')
    self.assertEqual(tree.body[0].names[2].name, 'c')