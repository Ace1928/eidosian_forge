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