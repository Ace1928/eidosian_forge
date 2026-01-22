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
class TreeAssertVisitor(VisitorTransform):

    def __init__(self):
        super(TreeAssertVisitor, self).__init__()
        self._module_pos = None
        self._c_patterns = []
        self._c_antipatterns = []

    def create_c_file_validator(self):
        patterns, antipatterns = (self._c_patterns, self._c_antipatterns)

        def fail(pos, pattern, found, file_path):
            Errors.error(pos, "Pattern '%s' %s found in %s" % (pattern, 'was' if found else 'was not', file_path))

        def extract_section(file_path, content, start, end):
            if start:
                split = re.search(start, content)
                if split:
                    content = content[split.end():]
                else:
                    fail(self._module_pos, start, found=False, file_path=file_path)
            if end:
                split = re.search(end, content)
                if split:
                    content = content[:split.start()]
                else:
                    fail(self._module_pos, end, found=False, file_path=file_path)
            return content

        def validate_file_content(file_path, content):
            for pattern in patterns:
                start, end, pattern = _parse_pattern(pattern)
                section = extract_section(file_path, content, start, end)
                if not re.search(pattern, section):
                    fail(self._module_pos, pattern, found=False, file_path=file_path)
            for antipattern in antipatterns:
                start, end, antipattern = _parse_pattern(antipattern)
                section = extract_section(file_path, content, start, end)
                if re.search(antipattern, section):
                    fail(self._module_pos, antipattern, found=True, file_path=file_path)

        def validate_c_file(result):
            c_file = result.c_file
            if not (patterns or antipatterns):
                return result
            with open(c_file, encoding='utf8') as f:
                content = f.read()
            content = _strip_c_comments(content)
            validate_file_content(c_file, content)
            html_file = os.path.splitext(c_file)[0] + '.html'
            if os.path.exists(html_file) and os.path.getmtime(c_file) <= os.path.getmtime(html_file):
                with open(html_file, encoding='utf8') as f:
                    content = f.read()
                content = _strip_cython_code_from_html(content)
                validate_file_content(html_file, content)
        return validate_c_file

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

    def visit_ModuleNode(self, node):
        self._module_pos = node.pos
        self._check_directives(node)
        self.visitchildren(node)
        return node

    def visit_CompilerDirectivesNode(self, node):
        self._check_directives(node)
        self.visitchildren(node)
        return node
    visit_Node = VisitorTransform.recurse_to_children