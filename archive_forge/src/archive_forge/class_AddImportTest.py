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
class AddImportTest(test_utils.TestCase):

    def test_add_normal_import(self):
        tree = ast.parse('')
        self.assertEqual('a.b.c', import_utils.add_import(tree, 'a.b.c', from_import=False))
        self.assertEqual('import a.b.c\n', pasta.dump(tree))

    def test_add_normal_import_with_asname(self):
        tree = ast.parse('')
        self.assertEqual('d', import_utils.add_import(tree, 'a.b.c', asname='d', from_import=False))
        self.assertEqual('import a.b.c as d\n', pasta.dump(tree))

    def test_add_from_import(self):
        tree = ast.parse('')
        self.assertEqual('c', import_utils.add_import(tree, 'a.b.c', from_import=True))
        self.assertEqual('from a.b import c\n', pasta.dump(tree))

    def test_add_from_import_with_asname(self):
        tree = ast.parse('')
        self.assertEqual('d', import_utils.add_import(tree, 'a.b.c', asname='d', from_import=True))
        self.assertEqual('from a.b import c as d\n', pasta.dump(tree))

    def test_add_single_name_from_import(self):
        tree = ast.parse('')
        self.assertEqual('foo', import_utils.add_import(tree, 'foo', from_import=True))
        self.assertEqual('import foo\n', pasta.dump(tree))

    def test_add_single_name_from_import_with_asname(self):
        tree = ast.parse('')
        self.assertEqual('bar', import_utils.add_import(tree, 'foo', asname='bar', from_import=True))
        self.assertEqual('import foo as bar\n', pasta.dump(tree))

    def test_add_existing_import(self):
        tree = ast.parse('from a.b import c')
        self.assertEqual('c', import_utils.add_import(tree, 'a.b.c'))
        self.assertEqual('from a.b import c\n', pasta.dump(tree))

    def test_add_existing_import_aliased(self):
        tree = ast.parse('from a.b import c as d')
        self.assertEqual('d', import_utils.add_import(tree, 'a.b.c'))
        self.assertEqual('from a.b import c as d\n', pasta.dump(tree))

    def test_add_existing_import_aliased_with_asname(self):
        tree = ast.parse('from a.b import c as d')
        self.assertEqual('d', import_utils.add_import(tree, 'a.b.c', asname='e'))
        self.assertEqual('from a.b import c as d\n', pasta.dump(tree))

    def test_add_existing_import_normal_import(self):
        tree = ast.parse('import a.b.c')
        self.assertEqual('a.b', import_utils.add_import(tree, 'a.b', from_import=False))
        self.assertEqual('import a.b.c\n', pasta.dump(tree))

    def test_add_existing_import_normal_import_aliased(self):
        tree = ast.parse('import a.b.c as d')
        self.assertEqual('a.b', import_utils.add_import(tree, 'a.b', from_import=False))
        self.assertEqual('d', import_utils.add_import(tree, 'a.b.c', from_import=False))
        self.assertEqual('import a.b\nimport a.b.c as d\n', pasta.dump(tree))

    def test_add_import_with_conflict(self):
        tree = ast.parse('def c(): pass\n')
        self.assertEqual('c_1', import_utils.add_import(tree, 'a.b.c', from_import=True))
        self.assertEqual('from a.b import c as c_1\ndef c():\n  pass\n', pasta.dump(tree))

    def test_add_import_with_asname_with_conflict(self):
        tree = ast.parse('def c(): pass\n')
        self.assertEqual('c_1', import_utils.add_import(tree, 'a.b', asname='c', from_import=True))
        self.assertEqual('from a import b as c_1\ndef c():\n  pass\n', pasta.dump(tree))

    def test_merge_from_import(self):
        tree = ast.parse('from a.b import c')
        self.assertEqual('x', import_utils.add_import(tree, 'a.b.x', merge_from_imports=False))
        self.assertEqual('from a.b import x\nfrom a.b import c\n', pasta.dump(tree))
        self.assertEqual('y', import_utils.add_import(tree, 'a.b.y', merge_from_imports=True))
        self.assertEqual('from a.b import x, y\nfrom a.b import c\n', pasta.dump(tree))

    def test_add_import_after_docstring(self):
        tree = ast.parse("'Docstring.'")
        self.assertEqual('a', import_utils.add_import(tree, 'a'))
        self.assertEqual("'Docstring.'\nimport a\n", pasta.dump(tree))