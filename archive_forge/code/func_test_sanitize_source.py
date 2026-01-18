from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pasta
from pasta.augment import errors
from pasta.base import ast_utils
from pasta.base import test_utils
def test_sanitize_source(self):
    coding_lines = ('# -*- coding: latin-1 -*-', '# -*- coding: iso-8859-15 -*-', '# vim: set fileencoding=ascii :', '# This Python file uses the following encoding: utf-8')
    src_template = '{coding}\na = 123\n'
    sanitized_src = '# (removed coding)\na = 123\n'
    for line in coding_lines:
        src = src_template.format(coding=line)
        self.assertEqual(sanitized_src, ast_utils.sanitize_source(src))
        src_prefix = '"""Docstring."""\n'
        self.assertEqual(src_prefix + sanitized_src, ast_utils.sanitize_source(src_prefix + src))
        src_prefix = '"""Docstring."""\n# line 2\n'
        self.assertEqual(src_prefix + src, ast_utils.sanitize_source(src_prefix + src))