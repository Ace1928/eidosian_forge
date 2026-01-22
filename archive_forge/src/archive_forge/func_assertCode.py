from __future__ import absolute_import
import os
import re
import unittest
import shlex
import sys
import tempfile
import textwrap
from io import open
from functools import partial
from .Compiler import Errors
from .CodeWriter import CodeWriter
from .Compiler.TreeFragment import TreeFragment, strip_common_indent
from .Compiler.Visitor import TreeVisitor, VisitorTransform
from .Compiler import TreePath
def assertCode(self, expected, result_tree):
    result_lines = self.codeToLines(result_tree)
    expected_lines = strip_common_indent(expected.split('\n'))
    for idx, (line, expected_line) in enumerate(zip(result_lines, expected_lines)):
        self.assertEqual(expected_line, line, 'Line %d:\nGot: %s\nExp: %s' % (idx, line, expected_line))
    self.assertEqual(len(result_lines), len(expected_lines), 'Unmatched lines. Got:\n%s\nExpected:\n%s' % ('\n'.join(result_lines), expected))