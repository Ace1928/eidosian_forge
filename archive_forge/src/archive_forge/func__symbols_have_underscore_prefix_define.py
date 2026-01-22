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
def _symbols_have_underscore_prefix_define(self, env: 'Environment') -> T.Optional[bool]:
    """
        Check if symbols have underscore prefix by querying the
        __USER_LABEL_PREFIX__ define that most compilers provide
        for this. Return if functions have underscore prefix or None
        if it was not possible to determine, like when the compiler
        does not set the define or the define has an unexpected value.
        """
    delim = '"MESON_HAVE_UNDERSCORE_DELIMITER" '
    code = f'\n        #ifndef __USER_LABEL_PREFIX__\n        #define MESON_UNDERSCORE_PREFIX unsupported\n        #else\n        #define MESON_UNDERSCORE_PREFIX __USER_LABEL_PREFIX__\n        #endif\n        {delim}MESON_UNDERSCORE_PREFIX\n        '
    with self._build_wrapper(code, env, mode=CompileCheckMode.PREPROCESS, want_output=False) as p:
        if p.returncode != 0:
            raise RuntimeError(f'BUG: Unable to preprocess _symbols_have_underscore_prefix_define check: {p.stdout}')
        symbol_prefix = p.stdout.partition(delim)[-1].rstrip()
        mlog.debug(f'Queried compiler for function prefix: __USER_LABEL_PREFIX__ is "{symbol_prefix!s}"')
        if symbol_prefix == '_':
            return True
        elif symbol_prefix == '':
            return False
        else:
            return None