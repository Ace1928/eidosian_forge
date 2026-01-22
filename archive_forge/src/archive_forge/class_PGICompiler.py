from __future__ import annotations
import typing as T
import os
from pathlib import Path
from ..compilers import clike_debug_args, clike_optimization_args
from ...mesonlib import OptionKey
class PGICompiler(Compiler):
    id = 'pgi'

    def __init__(self) -> None:
        self.base_options = {OptionKey('b_pch')}
        default_warn_args = ['-Minform=inform']
        self.warn_args: T.Dict[str, T.List[str]] = {'0': [], '1': default_warn_args, '2': default_warn_args, '3': default_warn_args, 'everything': default_warn_args}

    def get_module_incdir_args(self) -> T.Tuple[str]:
        return ('-module',)

    def gen_import_library_args(self, implibname: str) -> T.List[str]:
        return []

    def get_pic_args(self) -> T.List[str]:
        if self.info.is_linux():
            return ['-fPIC']
        return []

    def openmp_flags(self) -> T.List[str]:
        return ['-mp']

    def get_optimization_args(self, optimization_level: str) -> T.List[str]:
        return clike_optimization_args[optimization_level]

    def get_debug_args(self, is_debug: bool) -> T.List[str]:
        return clike_debug_args[is_debug]

    def compute_parameters_with_absolute_paths(self, parameter_list: T.List[str], build_dir: str) -> T.List[str]:
        for idx, i in enumerate(parameter_list):
            if i[:2] == '-I' or i[:2] == '-L':
                parameter_list[idx] = i[:2] + os.path.normpath(os.path.join(build_dir, i[2:]))
        return parameter_list

    def get_always_args(self) -> T.List[str]:
        return []

    def get_pch_suffix(self) -> str:
        return 'pch'

    def get_pch_use_args(self, pch_dir: str, header: str) -> T.List[str]:
        hdr = Path(pch_dir).resolve().parent / header
        if self.language == 'cpp':
            return ['--pch', '--pch_dir', str(hdr.parent), f'-I{hdr.parent}']
        else:
            return []

    def thread_flags(self, env: 'Environment') -> T.List[str]:
        return []