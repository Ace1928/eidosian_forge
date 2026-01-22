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
class ClangClCCompiler(_ClangCStds, ClangClCompiler, VisualStudioLikeCCompilerMixin, CCompiler):

    def __init__(self, exelist: T.List[str], version: str, for_machine: MachineChoice, is_cross: bool, info: 'MachineInfo', target: str, exe_wrapper: T.Optional['ExternalProgram']=None, linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None):
        CCompiler.__init__(self, [], exelist, version, for_machine, is_cross, info, exe_wrapper, linker=linker, full_version=full_version)
        ClangClCompiler.__init__(self, target)

    def get_option_compile_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        key = OptionKey('std', machine=self.for_machine, lang=self.language)
        std = options[key].value
        if std != 'none':
            return [f'/clang:-std={std}']
        return []