from __future__ import annotations
import collections
import functools
import glob
import itertools
import os
import re
import subprocess
import copy
import typing as T
from pathlib import Path
from ... import arglist
from ... import mesonlib
from ... import mlog
from ...linkers.linkers import GnuLikeDynamicLinkerMixin, SolarisDynamicLinker, CompCertDynamicLinker
from ...mesonlib import LibType, OptionKey
from .. import compilers
from ..compilers import CompileCheckMode
from .visualstudio import VisualStudioLikeCompiler
def _symbols_have_underscore_prefix_searchbin(self, env: 'Environment') -> bool:
    """
        Check if symbols have underscore prefix by compiling a small test binary
        and then searching the binary for the string,
        """
    symbol_name = b'meson_uscore_prefix'
    code = '#ifdef __cplusplus\n        extern "C" {\n        #endif\n        void ' + symbol_name.decode() + ' (void) {}\n        #ifdef __cplusplus\n        }\n        #endif\n        '
    args = self.get_compiler_check_args(CompileCheckMode.COMPILE)
    n = '_symbols_have_underscore_prefix_searchbin'
    with self._build_wrapper(code, env, extra_args=args, mode=CompileCheckMode.COMPILE, want_output=True) as p:
        if p.returncode != 0:
            raise RuntimeError(f'BUG: Unable to compile {n!r} check: {p.stderr}')
        if not os.path.isfile(p.output_name):
            raise RuntimeError(f"BUG: Can't find compiled test code for {n!r} check")
        with open(p.output_name, 'rb') as o:
            for line in o:
                if b'_' + symbol_name in line:
                    mlog.debug('Underscore prefix check found prefixed function in binary')
                    return True
                elif symbol_name in line:
                    mlog.debug('Underscore prefix check found non-prefixed function in binary')
                    return False
    raise RuntimeError(f'BUG: {n!r} check did not find symbol string in binary')