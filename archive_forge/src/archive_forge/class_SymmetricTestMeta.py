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
class SymmetricTestMeta(type):

    def __new__(mcs, name, bases, inst_dict):

        def symmetric_test_generator(filepath):

            def test(self):
                with open(filepath, 'r') as handle:
                    src = handle.read()
                t = ast_utils.parse(src)
                annotator = annotate.AstAnnotator(src)
                annotator.visit(t)
                self.assertMultiLineEqual(codegen.to_str(t), src)
                self.assertEqual([], annotator.tokens._parens, 'Unmatched parens')
            return test
        test_method_prefix = 'test_symmetric_'
        data_dir = os.path.join(TESTDATA_DIR, 'ast')
        for dirpath, dirs, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith('.in'):
                    full_path = os.path.join(dirpath, filename)
                    inst_dict[test_method_prefix + filename[:-3]] = unittest.skipIf(not _is_syntax_valid(full_path), 'Test contains syntax not supported by this version.')(symmetric_test_generator(full_path))
        return type.__new__(mcs, name, bases, inst_dict)