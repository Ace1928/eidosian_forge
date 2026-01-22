from __future__ import annotations
import os
import typing as T
from ..mesonlib import EnvironmentException, OptionKey, get_meson_command
from .compilers import Compiler
from .mixins.metrowerks import MetrowerksCompiler, mwasmarm_instruction_set_args, mwasmeppc_instruction_set_args
class MasmARMCompiler(Compiler):
    language = 'masm'
    id = 'armasm'

    def needs_static_linker(self) -> bool:
        return True

    def get_always_args(self) -> T.List[str]:
        return ['-nologo']

    def get_werror_args(self) -> T.List[str]:
        return []

    def get_output_args(self, outputname: str) -> T.List[str]:
        return ['-o', outputname]

    def get_optimization_args(self, optimization_level: str) -> T.List[str]:
        return []

    def get_debug_args(self, is_debug: bool) -> T.List[str]:
        if is_debug:
            return ['-g']
        return []

    def sanity_check(self, work_dir: str, environment: 'Environment') -> None:
        if self.info.cpu_family not in {'arm', 'aarch64'}:
            raise EnvironmentException(f'ASM compiler {self.id!r} does not support {self.info.cpu_family} CPU family')

    def get_pic_args(self) -> T.List[str]:
        return []

    def get_include_args(self, path: str, is_system: bool) -> T.List[str]:
        if not path:
            path = '.'
        return ['-i' + path]

    def compute_parameters_with_absolute_paths(self, parameter_list: T.List[str], build_dir: str) -> T.List[str]:
        for idx, i in enumerate(parameter_list):
            if i[:2] == '-I':
                parameter_list[idx] = i[:2] + os.path.normpath(os.path.join(build_dir, i[2:]))
        return parameter_list

    def get_crt_compile_args(self, crt_val: str, buildtype: str) -> T.List[str]:
        return []

    def depfile_for_object(self, objfile: str) -> T.Optional[str]:
        return None