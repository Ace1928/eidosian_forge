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
def _check_directives(self, node):
    directives = node.directives
    if 'test_assert_path_exists' in directives:
        for path in directives['test_assert_path_exists']:
            if TreePath.find_first(node, path) is None:
                Errors.error(node.pos, "Expected path '%s' not found in result tree" % path)
    if 'test_fail_if_path_exists' in directives:
        for path in directives['test_fail_if_path_exists']:
            first_node = TreePath.find_first(node, path)
            if first_node is not None:
                Errors.error(first_node.pos, "Unexpected path '%s' found in result tree" % path)
    if 'test_assert_c_code_has' in directives:
        self._c_patterns.extend(directives['test_assert_c_code_has'])
    if 'test_fail_if_c_code_has' in directives:
        self._c_antipatterns.extend(directives['test_fail_if_c_code_has'])