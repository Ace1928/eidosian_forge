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
def unpack_source_tree(tree_file, workdir, cython_root):
    programs = {'PYTHON': [sys.executable], 'CYTHON': [sys.executable, os.path.join(cython_root, 'cython.py')], 'CYTHONIZE': [sys.executable, os.path.join(cython_root, 'cythonize.py')]}
    if workdir is None:
        workdir = tempfile.mkdtemp()
    header, cur_file = ([], None)
    with open(tree_file, 'rb') as f:
        try:
            for line in f:
                if line[:5] == b'#####':
                    filename = line.strip().strip(b'#').strip().decode('utf8').replace('/', os.path.sep)
                    path = os.path.join(workdir, filename)
                    if not os.path.exists(os.path.dirname(path)):
                        os.makedirs(os.path.dirname(path))
                    if cur_file is not None:
                        to_close, cur_file = (cur_file, None)
                        to_close.close()
                    cur_file = open(path, 'wb')
                elif cur_file is not None:
                    cur_file.write(line)
                elif line.strip() and (not line.lstrip().startswith(b'#')):
                    if line.strip() not in (b'"""', b"'''"):
                        command = shlex.split(line.decode('utf8'))
                        if not command:
                            continue
                        prog, args = (command[0], command[1:])
                        try:
                            header.append(programs[prog] + args)
                        except KeyError:
                            header.append(command)
        finally:
            if cur_file is not None:
                cur_file.close()
    return (workdir, header)