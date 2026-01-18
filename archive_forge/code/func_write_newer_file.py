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
def write_newer_file(file_path, newer_than, content, dedent=False, encoding=None):
    """
    Write `content` to the file `file_path` without translating `'\\n'`
    into `os.linesep` and make sure it is newer than the file `newer_than`.

    The default encoding is `'utf-8'` (same as for `write_file`).
    """
    write_file(file_path, content, dedent=dedent, encoding=encoding)
    try:
        other_time = os.path.getmtime(newer_than)
    except OSError:
        other_time = None
    while other_time is None or other_time >= os.path.getmtime(file_path):
        write_file(file_path, content, dedent=dedent, encoding=encoding)