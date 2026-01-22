from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import difflib
import itertools
import os.path
from six import with_metaclass
import sys
import textwrap
import unittest
import pasta
from pasta.base import annotate
from pasta.base import ast_utils
from pasta.base import codegen
from pasta.base import formatting as fmt
from pasta.base import test_utils
class FstringTest(test_utils.TestCase):
    """Tests fstring support more in-depth."""

    @test_utils.requires_features('fstring')
    def test_fstring(self):
        src = 'f"a {b} c d {e}"'
        t = pasta.parse(src)
        node = t.body[0].value
        self.assertEqual(fmt.get(node, 'content'), 'f"a {__pasta_fstring_val_0__} c d {__pasta_fstring_val_1__}"')

    @test_utils.requires_features('fstring')
    def test_fstring_escaping(self):
        src = 'f"a {{{b} {{c}}"'
        t = pasta.parse(src)
        node = t.body[0].value
        self.assertEqual(fmt.get(node, 'content'), 'f"a {{{__pasta_fstring_val_0__} {{c}}"')