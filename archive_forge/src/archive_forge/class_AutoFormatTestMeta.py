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
class AutoFormatTestMeta(type):

    def __new__(mcs, name, bases, inst_dict):

        def auto_format_test_generator(input_file):

            def test(self):
                with open(input_file, 'r') as handle:
                    src = handle.read()
                t = ast.parse(src)
                auto_formatted = codegen.to_str(t)
                self.assertMultiLineEqual(src, auto_formatted)
            return test
        test_method_prefix = 'test_auto_format_'
        data_dir = os.path.join(TESTDATA_DIR, 'codegen')
        for dirpath, _, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith('.in'):
                    full_path = os.path.join(dirpath, filename)
                    inst_dict[test_method_prefix + filename[:-3]] = unittest.skipIf(not _is_syntax_valid(full_path), 'Test contains syntax not supported by this version.')(auto_format_test_generator(full_path))
        return type.__new__(mcs, name, bases, inst_dict)