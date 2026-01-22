from __future__ import annotations
import enum
import os.path
import string
import typing as T
from .. import coredata
from .. import mlog
from ..mesonlib import (
from .compilers import Compiler
def _to_host_flags(self, flags: T.List[str], phase: _Phase=_Phase.COMPILER) -> T.List[str]:
    """
        Translate generic "GCC-speak" plus particular "NVCC-speak" flags to NVCC flags.

        NVCC's "short" flags have broad similarities to the GCC standard, but have
        gratuitous, irritating differences.
        """
    xflags = []
    flagit = iter(flags)
    for flag in flagit:
        if flag in self._FLAG_PASSTHRU_NOARGS:
            xflags.append(flag)
            continue
        if flag[:1] not in '-/':
            xflags.append(flag)
            continue
        elif flag[:1] == '/':
            wrap = '"' if ',' in flag else ''
            xflags.append(f'-X{phase.value}={wrap}{flag}{wrap}')
            continue
        elif len(flag) >= 2 and flag[0] == '-' and (flag[1] in 'IDULlmOxmte'):
            if flag[2:3] == '':
                try:
                    val = next(flagit)
                except StopIteration:
                    pass
            elif flag[2:3] == '=':
                val = flag[3:]
            else:
                val = flag[2:]
            flag = flag[:2]
        elif flag in self._FLAG_LONG2SHORT_WITHARGS or flag in self._FLAG_SHORT2LONG_WITHARGS:
            try:
                val = next(flagit)
            except StopIteration:
                pass
        elif flag.split('=', 1)[0] in self._FLAG_LONG2SHORT_WITHARGS or flag.split('=', 1)[0] in self._FLAG_SHORT2LONG_WITHARGS:
            flag, val = flag.split('=', 1)
        elif flag.startswith('-isystem'):
            val = flag[8:].strip()
            flag = flag[:8]
        else:
            if flag == '-ffast-math':
                xflags.append('-use_fast_math')
                xflags.append('-Xcompiler=' + flag)
            elif flag == '-fno-fast-math':
                xflags.append('-ftz=false')
                xflags.append('-prec-div=true')
                xflags.append('-prec-sqrt=true')
                xflags.append('-Xcompiler=' + flag)
            elif flag == '-freciprocal-math':
                xflags.append('-prec-div=false')
                xflags.append('-Xcompiler=' + flag)
            elif flag == '-fno-reciprocal-math':
                xflags.append('-prec-div=true')
                xflags.append('-Xcompiler=' + flag)
            else:
                xflags.append('-Xcompiler=' + self._shield_nvcc_list_arg(flag))
            continue
        assert val is not None
        flag = self._FLAG_LONG2SHORT_WITHARGS.get(flag, flag)
        if flag in {'-include', '-isystem', '-I', '-L', '-l'}:
            if len(flag) == 2:
                xflags.append(flag + self._shield_nvcc_list_arg(val))
            elif flag == '-isystem' and val in self.host_compiler.get_default_include_dirs():
                pass
            else:
                xflags.append(flag)
                xflags.append(self._shield_nvcc_list_arg(val))
        elif flag == '-O':
            if val == 'fast':
                xflags.append('-O3')
                xflags.append('-use_fast_math')
                xflags.append('-Xcompiler')
                xflags.append(flag + val)
            elif val in {'s', 'g', 'z'}:
                xflags.append('-Xcompiler')
                xflags.append(flag + val)
            else:
                xflags.append(flag + val)
        elif flag in {'-D', '-U', '-m', '-t'}:
            xflags.append(flag + val)
        elif flag in {'-std'}:
            xflags.append(flag + '=' + val)
        else:
            xflags.append(flag)
            xflags.append(val)
    return self._merge_flags(xflags)