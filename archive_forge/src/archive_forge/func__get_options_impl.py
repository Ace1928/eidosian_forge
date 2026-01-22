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
def _get_options_impl(self, opts: 'MutableKeyedOptionDictType', cpp_stds: T.List[str]) -> 'MutableKeyedOptionDictType':
    key = OptionKey('std', machine=self.for_machine, lang=self.language)
    opts.update({key.evolve('eh'): coredata.UserComboOption('C++ exception handling type.', ['none', 'default', 'a', 's', 'sc'], 'default'), key.evolve('rtti'): coredata.UserBooleanOption('Enable RTTI', True), key.evolve('winlibs'): coredata.UserArrayOption('Windows libs to link against.', msvc_winlibs)})
    std_opt = opts[key]
    assert isinstance(std_opt, coredata.UserStdOption), 'for mypy'
    std_opt.set_versions(cpp_stds)
    return opts