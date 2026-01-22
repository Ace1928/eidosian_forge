from __future__ import annotations
import typing as T
import os
from .. import coredata
from .compilers import (
from .mixins.clike import CLikeCompiler
from .mixins.gnu import GnuCompiler,  gnu_optimization_args
from .mixins.intel import IntelGnuLikeCompiler, IntelVisualStudioLikeCompiler
from .mixins.clang import ClangCompiler
from .mixins.elbrus import ElbrusCompiler
from .mixins.pgi import PGICompiler
from mesonbuild.mesonlib import (
class FlangFortranCompiler(ClangCompiler, FortranCompiler):
    id = 'flang'

    def __init__(self, exelist: T.List[str], version: str, for_machine: MachineChoice, is_cross: bool, info: 'MachineInfo', exe_wrapper: T.Optional['ExternalProgram']=None, linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None):
        FortranCompiler.__init__(self, exelist, version, for_machine, is_cross, info, exe_wrapper, linker=linker, full_version=full_version)
        ClangCompiler.__init__(self, {})
        default_warn_args = ['-Minform=inform']
        self.warn_args = {'0': [], '1': default_warn_args, '2': default_warn_args, '3': default_warn_args, 'everything': default_warn_args}

    def language_stdlib_only_link_flags(self, env: 'Environment') -> T.List[str]:
        search_dirs: T.List[str] = []
        for d in self.get_compiler_dirs(env, 'libraries'):
            search_dirs.append(f'-L{d}')
        return search_dirs + ['-lflang', '-lpgmath']