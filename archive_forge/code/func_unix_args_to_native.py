from __future__ import annotations
import abc
import os
import typing as T
from ... import arglist
from ... import mesonlib
from ... import mlog
from mesonbuild.compilers.compilers import CompileCheckMode
@classmethod
def unix_args_to_native(cls, args: T.List[str]) -> T.List[str]:
    result: T.List[str] = []
    for i in args:
        if i in {'-mms-bitfields', '-pthread'}:
            continue
        if i.startswith('-LIBPATH:'):
            i = '/LIBPATH:' + i[9:]
        elif i.startswith('-L'):
            i = '/LIBPATH:' + i[2:]
        elif i.startswith('-l'):
            name = i[2:]
            if name in cls.ignore_libs:
                continue
            else:
                i = name + '.lib'
        elif i.startswith('-isystem'):
            if i.startswith('-isystem='):
                i = '/I' + i[9:]
            else:
                i = '/I' + i[8:]
        elif i.startswith('-idirafter'):
            if i.startswith('-idirafter='):
                i = '/I' + i[11:]
            else:
                i = '/I' + i[10:]
        elif i == '-pthread':
            continue
        elif i.startswith('/source-charset:') or i.startswith('/execution-charset:') or i == '/validate-charset-':
            try:
                result.remove('/utf-8')
            except ValueError:
                pass
        result.append(i)
    return result