from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import os.path
import unittest
from six import with_metaclass
import pasta
from pasta.base import codegen
from pasta.base import test_utils
class AutoFormatTest(with_metaclass(AutoFormatTestMeta, test_utils.TestCase)):
    """Tests that code without formatting info is printed neatly."""

    def test_imports(self):
        src = 'from a import b\nimport c, d\nfrom ..e import f, g\n'
        t = ast.parse(src)
        self.assertEqual(src, pasta.dump(t))

    @test_utils.requires_features('exec_node')
    def test_exec_node_default(self):
        src = 'exec foo in bar'
        t = ast.parse(src)
        self.assertEqual('exec(foo, bar)\n', pasta.dump(t))

    @test_utils.requires_features('bytes_node')
    def test_bytes(self):
        src = "b'foo'"
        t = ast.parse(src)
        self.assertEqual("b'foo'\n", pasta.dump(t))

    def test_default_indentation(self):
        for indent in ('  ', '    ', '\t'):
            src = 'def a():\n' + indent + 'b\n'
            t = pasta.parse(src)
            t.body.extend(ast.parse('def c(): d').body)
            self.assertEqual(codegen.to_str(t), src + 'def c():\n' + indent + 'd\n')