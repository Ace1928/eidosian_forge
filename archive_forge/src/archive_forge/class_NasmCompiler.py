from __future__ import annotations
import os
import typing as T
from ..mesonlib import EnvironmentException, OptionKey, get_meson_command
from .compilers import Compiler
from .mixins.metrowerks import MetrowerksCompiler, mwasmarm_instruction_set_args, mwasmeppc_instruction_set_args
class NasmCompiler(Compiler):
    language = 'nasm'
    id = 'nasm'
    crt_args: T.Dict[str, T.List[str]] = {'none': [], 'md': ['/DEFAULTLIB:ucrt.lib', '/DEFAULTLIB:vcruntime.lib', '/DEFAULTLIB:msvcrt.lib'], 'mdd': ['/DEFAULTLIB:ucrtd.lib', '/DEFAULTLIB:vcruntimed.lib', '/DEFAULTLIB:msvcrtd.lib'], 'mt': ['/DEFAULTLIB:libucrt.lib', '/DEFAULTLIB:libvcruntime.lib', '/DEFAULTLIB:libcmt.lib'], 'mtd': ['/DEFAULTLIB:libucrtd.lib', '/DEFAULTLIB:libvcruntimed.lib', '/DEFAULTLIB:libcmtd.lib']}

    def __init__(self, ccache: T.List[str], exelist: T.List[str], version: str, for_machine: 'MachineChoice', info: 'MachineInfo', linker: T.Optional['DynamicLinker']=None, full_version: T.Optional[str]=None, is_cross: bool=False):
        super().__init__(ccache, exelist, version, for_machine, info, linker, full_version, is_cross)
        if 'link' in self.linker.id:
            self.base_options.add(OptionKey('b_vscrt'))

    def needs_static_linker(self) -> bool:
        return True

    def get_always_args(self) -> T.List[str]:
        cpu = '64' if self.info.is_64_bit else '32'
        if self.info.is_windows() or self.info.is_cygwin():
            plat = 'win'
            define = f'WIN{cpu}'
        elif self.info.is_darwin():
            plat = 'macho'
            define = 'MACHO'
        else:
            plat = 'elf'
            define = 'ELF'
        args = ['-f', f'{plat}{cpu}', f'-D{define}']
        if self.info.is_64_bit:
            args.append('-D__x86_64__')
        return args

    def get_werror_args(self) -> T.List[str]:
        return ['-Werror']

    def get_output_args(self, outputname: str) -> T.List[str]:
        return ['-o', outputname]

    def unix_args_to_native(self, args: T.List[str]) -> T.List[str]:
        outargs: T.List[str] = []
        for arg in args:
            if arg == '-pthread':
                continue
            outargs.append(arg)
        return outargs

    def get_optimization_args(self, optimization_level: str) -> T.List[str]:
        return nasm_optimization_args[optimization_level]

    def get_debug_args(self, is_debug: bool) -> T.List[str]:
        if is_debug:
            if self.info.is_windows():
                return []
            return ['-g', '-F', 'dwarf']
        return []

    def get_depfile_suffix(self) -> str:
        return 'd'

    def get_dependency_gen_args(self, outtarget: str, outfile: str) -> T.List[str]:
        return ['-MD', outfile, '-MQ', outtarget]

    def sanity_check(self, work_dir: str, environment: 'Environment') -> None:
        if self.info.cpu_family not in {'x86', 'x86_64'}:
            raise EnvironmentException(f'ASM compiler {self.id!r} does not support {self.info.cpu_family} CPU family')

    def get_pic_args(self) -> T.List[str]:
        return []

    def get_include_args(self, path: str, is_system: bool) -> T.List[str]:
        if not path:
            path = '.'
        return ['-I' + path]

    def compute_parameters_with_absolute_paths(self, parameter_list: T.List[str], build_dir: str) -> T.List[str]:
        for idx, i in enumerate(parameter_list):
            if i[:2] == '-I':
                parameter_list[idx] = i[:2] + os.path.normpath(os.path.join(build_dir, i[2:]))
        return parameter_list

    def get_crt_compile_args(self, crt_val: str, buildtype: str) -> T.List[str]:
        return []

    def get_crt_link_args(self, crt_val: str, buildtype: str) -> T.List[str]:
        if not self.info.is_windows():
            return []
        return self.crt_args[self.get_crt_val(crt_val, buildtype)]