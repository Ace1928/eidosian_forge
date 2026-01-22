from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import os
import datetime
import traceback
import platform  # NOQA
from _ast import *  # NOQA
from ast import parse  # NOQA
from setuptools import setup, Extension, Distribution  # NOQA
from setuptools.command import install_lib  # NOQA
from setuptools.command.sdist import sdist as _sdist  # NOQA
def _package_data(fn):
    data = {}
    with open(fn, **open_kw) as fp:
        parsing = False
        lines = []
        for line in fp.readlines():
            if sys.version_info < (3,):
                line = line.decode('utf-8')
            if line.startswith('_package_data'):
                if 'dict(' in line:
                    parsing = 'python'
                    lines.append('dict(\n')
                elif line.endswith('= {\n'):
                    parsing = 'python'
                    lines.append('{\n')
                else:
                    raise NotImplementedError
                continue
            if not parsing:
                continue
            if parsing == 'python':
                if line.startswith(')') or line.startswith('}'):
                    lines.append(line)
                    try:
                        data = literal_eval(''.join(lines))
                    except SyntaxError as e:
                        context = 2
                        from_line = e.lineno - (context + 1)
                        to_line = e.lineno + (context - 1)
                        w = len(str(to_line))
                        for index, line in enumerate(lines):
                            if from_line <= index <= to_line:
                                print('{0:{1}}: {2}'.format(index, w, line).encode('utf-8'), end='')
                                if index == e.lineno - 1:
                                    print('{0:{1}}  {2}^--- {3}'.format(' ', w, ' ' * e.offset, e.node))
                        raise
                    break
                lines.append(line)
            else:
                raise NotImplementedError
    return data