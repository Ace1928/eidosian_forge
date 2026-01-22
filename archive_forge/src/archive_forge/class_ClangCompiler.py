from __future__ import annotations
import os
import shutil
import typing as T
from ... import mesonlib
from ...linkers.linkers import AppleDynamicLinker, ClangClDynamicLinker, LLVMDynamicLinker, GnuGoldDynamicLinker, \
from ...mesonlib import OptionKey
from ..compilers import CompileCheckMode
from .gnu import GnuLikeCompiler
class ClangCompiler(GnuLikeCompiler):
    id = 'clang'

    def __init__(self, defines: T.Optional[T.Dict[str, str]]):
        super().__init__()
        self.defines = defines or {}
        self.base_options.update({OptionKey('b_colorout'), OptionKey('b_lto_threads'), OptionKey('b_lto_mode'), OptionKey('b_thinlto_cache'), OptionKey('b_thinlto_cache_dir')})
        if isinstance(self.linker, AppleDynamicLinker):
            self.base_options.add(OptionKey('b_bitcode'))
        self.can_compile_suffixes.add('ll')

    def get_colorout_args(self, colortype: str) -> T.List[str]:
        return clang_color_args[colortype][:]

    def has_builtin_define(self, define: str) -> bool:
        return define in self.defines

    def get_builtin_define(self, define: str) -> T.Optional[str]:
        return self.defines.get(define)

    def get_optimization_args(self, optimization_level: str) -> T.List[str]:
        return clang_optimization_args[optimization_level]

    def get_pch_suffix(self) -> str:
        return 'pch'

    def get_pch_use_args(self, pch_dir: str, header: str) -> T.List[str]:
        return ['-include-pch', os.path.join(pch_dir, self.get_pch_name(header))]

    def get_compiler_check_args(self, mode: CompileCheckMode) -> T.List[str]:
        myargs: T.List[str] = ['-Werror=implicit-function-declaration']
        if mode is CompileCheckMode.COMPILE:
            myargs.extend(['-Werror=unknown-warning-option', '-Werror=unused-command-line-argument'])
            if mesonlib.version_compare(self.version, '>=3.6.0'):
                myargs.append('-Werror=ignored-optimization-argument')
        return super().get_compiler_check_args(mode) + myargs

    def has_function(self, funcname: str, prefix: str, env: 'Environment', *, extra_args: T.Optional[T.List[str]]=None, dependencies: T.Optional[T.List['Dependency']]=None) -> T.Tuple[bool, bool]:
        if extra_args is None:
            extra_args = []
        if isinstance(self.linker, AppleDynamicLinker) and mesonlib.version_compare(self.version, '>=8.0'):
            extra_args.append('-Wl,-no_weak_imports')
        return super().has_function(funcname, prefix, env, extra_args=extra_args, dependencies=dependencies)

    def openmp_flags(self) -> T.List[str]:
        if mesonlib.version_compare(self.version, '>=3.8.0'):
            return ['-fopenmp']
        elif mesonlib.version_compare(self.version, '>=3.7.0'):
            return ['-fopenmp=libomp']
        else:
            return []

    @classmethod
    def use_linker_args(cls, linker: str, version: str) -> T.List[str]:
        if linker == 'qcld':
            return ['-fuse-ld=qcld']
        if linker == 'mold':
            return ['-fuse-ld=mold']
        if shutil.which(linker):
            if not shutil.which(linker):
                raise mesonlib.MesonException(f'Cannot find linker {linker}.')
            return [f'-fuse-ld={linker}']
        return super().use_linker_args(linker, version)

    def get_has_func_attribute_extra_args(self, name: str) -> T.List[str]:
        return ['-Werror=attributes']

    def get_coverage_link_args(self) -> T.List[str]:
        return ['--coverage']

    def get_lto_compile_args(self, *, threads: int=0, mode: str='default') -> T.List[str]:
        args: T.List[str] = []
        if mode == 'thin':
            if isinstance(self.linker, MoldDynamicLinker):
                if not mesonlib.version_compare(self.linker.version, '>=1.1'):
                    raise mesonlib.MesonException("LLVM's ThinLTO requires mold 1.1+")
            elif not isinstance(self.linker, (AppleDynamicLinker, ClangClDynamicLinker, LLVMDynamicLinker, GnuGoldDynamicLinker)):
                raise mesonlib.MesonException(f"LLVM's ThinLTO only works with gold, lld, lld-link, ld64 or mold, not {self.linker.id}")
            args.append(f'-flto={mode}')
        else:
            assert mode == 'default', 'someone forgot to wire something up'
            args.extend(super().get_lto_compile_args(threads=threads))
        return args

    def get_lto_link_args(self, *, threads: int=0, mode: str='default', thinlto_cache_dir: T.Optional[str]=None) -> T.List[str]:
        args = self.get_lto_compile_args(threads=threads, mode=mode)
        if mode == 'thin' and thinlto_cache_dir is not None:
            args.extend(self.linker.get_thinlto_cache_args(thinlto_cache_dir))
        if threads > 0:
            if not mesonlib.version_compare(self.version, '>=4.0.0'):
                raise mesonlib.MesonException('clang support for LTO threads requires clang >=4.0')
            args.append(f'-flto-jobs={threads}')
        return args