from __future__ import annotations
import os.path
import typing as T
from .. import coredata
from .. import mlog
from ..mesonlib import MesonException, version_compare, OptionKey
from .c_function_attributes import C_FUNC_ATTRIBUTES
from .mixins.clike import CLikeCompiler
from .mixins.ccrx import CcrxCompiler
from .mixins.xc16 import Xc16Compiler
from .mixins.compcert import CompCertCompiler
from .mixins.ti import TICompiler
from .mixins.arm import ArmCompiler, ArmclangCompiler
from .mixins.visualstudio import MSVCCompiler, ClangClCompiler
from .mixins.gnu import GnuCompiler
from .mixins.gnu import gnu_common_warning_args, gnu_c_warning_args
from .mixins.intel import IntelGnuLikeCompiler, IntelVisualStudioLikeCompiler
from .mixins.clang import ClangCompiler
from .mixins.elbrus import ElbrusCompiler
from .mixins.pgi import PGICompiler
from .mixins.emscripten import EmscriptenMixin
from .mixins.metrowerks import MetrowerksCompiler
from .mixins.metrowerks import mwccarm_instruction_set_args, mwcceppc_instruction_set_args
from .compilers import (
class _ClangCStds(CompilerMixinBase):
    """Mixin class for clang based compilers for setting C standards.

    This is used by both ClangCCompiler and ClangClCompiler, as they share
    the same versions
    """
    _C17_VERSION = '>=6.0.0'
    _C18_VERSION = '>=8.0.0'
    _C2X_VERSION = '>=9.0.0'
    _C23_VERSION = '>=18.0.0'

    def get_options(self) -> 'MutableKeyedOptionDictType':
        opts = super().get_options()
        stds = ['c89', 'c99', 'c11']
        if version_compare(self.version, self._C17_VERSION):
            stds += ['c17']
        if version_compare(self.version, self._C18_VERSION):
            stds += ['c18']
        if version_compare(self.version, self._C2X_VERSION):
            stds += ['c2x']
        if version_compare(self.version, self._C23_VERSION):
            stds += ['c23']
        std_opt = opts[OptionKey('std', machine=self.for_machine, lang=self.language)]
        assert isinstance(std_opt, coredata.UserStdOption), 'for mypy'
        std_opt.set_versions(stds, gnu=True)
        return opts