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
class GnuCCompiler(GnuCompiler, CCompiler):
    _C18_VERSION = '>=8.0.0'
    _C2X_VERSION = '>=9.0.0'
    _C23_VERSION = '>=14.0.0'
    _INVALID_PCH_VERSION = '>=3.4.0'

    def __init__(self, ccache: T.List[str], exelist: T.List[str], version: str, for_machine: MachineChoice, is_cross: bool, info: 'MachineInfo', exe_wrapper: T.Optional['ExternalProgram']=None, linker: T.Optional['DynamicLinker']=None, defines: T.Optional[T.Dict[str, str]]=None, full_version: T.Optional[str]=None):
        CCompiler.__init__(self, ccache, exelist, version, for_machine, is_cross, info, exe_wrapper, linker=linker, full_version=full_version)
        GnuCompiler.__init__(self, defines)
        default_warn_args = ['-Wall']
        if version_compare(self.version, self._INVALID_PCH_VERSION):
            default_warn_args += ['-Winvalid-pch']
        self.warn_args = {'0': [], '1': default_warn_args, '2': default_warn_args + ['-Wextra'], '3': default_warn_args + ['-Wextra', '-Wpedantic'], 'everything': default_warn_args + ['-Wextra', '-Wpedantic'] + self.supported_warn_args(gnu_common_warning_args) + self.supported_warn_args(gnu_c_warning_args)}

    def get_options(self) -> 'MutableKeyedOptionDictType':
        opts = CCompiler.get_options(self)
        stds = ['c89', 'c99', 'c11']
        if version_compare(self.version, self._C18_VERSION):
            stds += ['c17', 'c18']
        if version_compare(self.version, self._C2X_VERSION):
            stds += ['c2x']
        if version_compare(self.version, self._C23_VERSION):
            stds += ['c23']
        key = OptionKey('std', machine=self.for_machine, lang=self.language)
        std_opt = opts[key]
        assert isinstance(std_opt, coredata.UserStdOption), 'for mypy'
        std_opt.set_versions(stds, gnu=True)
        if self.info.is_windows() or self.info.is_cygwin():
            opts.update({key.evolve('winlibs'): coredata.UserArrayOption('Standard Win libraries to link against', gnu_winlibs)})
        return opts

    def get_option_compile_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        args = []
        std = options[OptionKey('std', lang=self.language, machine=self.for_machine)]
        if std.value != 'none':
            args.append('-std=' + std.value)
        return args

    def get_option_link_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        if self.info.is_windows() or self.info.is_cygwin():
            libs: T.List[str] = options[OptionKey('winlibs', lang=self.language, machine=self.for_machine)].value.copy()
            assert isinstance(libs, list)
            for l in libs:
                assert isinstance(l, str)
            return libs
        return []

    def get_pch_use_args(self, pch_dir: str, header: str) -> T.List[str]:
        return ['-fpch-preprocess', '-include', os.path.basename(header)]