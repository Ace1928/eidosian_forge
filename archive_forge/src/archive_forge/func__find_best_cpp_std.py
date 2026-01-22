from __future__ import annotations
import copy
import functools
import os.path
import typing as T
from .. import coredata
from .. import mlog
from ..mesonlib import MesonException, version_compare, OptionKey
from .compilers import (
from .c_function_attributes import CXX_FUNC_ATTRIBUTES, C_FUNC_ATTRIBUTES
from .mixins.clike import CLikeCompiler
from .mixins.ccrx import CcrxCompiler
from .mixins.ti import TICompiler
from .mixins.arm import ArmCompiler, ArmclangCompiler
from .mixins.visualstudio import MSVCCompiler, ClangClCompiler
from .mixins.gnu import GnuCompiler, gnu_common_warning_args, gnu_cpp_warning_args
from .mixins.intel import IntelGnuLikeCompiler, IntelVisualStudioLikeCompiler
from .mixins.clang import ClangCompiler
from .mixins.elbrus import ElbrusCompiler
from .mixins.pgi import PGICompiler
from .mixins.emscripten import EmscriptenMixin
from .mixins.metrowerks import MetrowerksCompiler
from .mixins.metrowerks import mwccarm_instruction_set_args, mwcceppc_instruction_set_args
@functools.lru_cache()
def _find_best_cpp_std(self, cpp_std: str) -> str:
    CPP_FALLBACKS = {'c++11': 'c++0x', 'gnu++11': 'gnu++0x', 'c++14': 'c++1y', 'gnu++14': 'gnu++1y', 'c++17': 'c++1z', 'gnu++17': 'gnu++1z', 'c++20': 'c++2a', 'gnu++20': 'gnu++2a', 'c++23': 'c++2b', 'gnu++23': 'gnu++2b', 'c++26': 'c++2c', 'gnu++26': 'gnu++2c'}
    assert self.id in frozenset(['clang', 'lcc', 'gcc', 'emscripten', 'armltdclang', 'intel-llvm'])
    if cpp_std not in CPP_FALLBACKS:
        return '-std=' + cpp_std
    for i in (cpp_std, CPP_FALLBACKS[cpp_std]):
        cpp_std_value = '-std=' + i
        if self._test_cpp_std_arg(cpp_std_value):
            return cpp_std_value
    raise MesonException(f'C++ Compiler does not support -std={cpp_std}')