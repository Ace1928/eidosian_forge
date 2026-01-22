from __future__ import annotations
import re
import typing as T
from ..mesonlib import listify, version_compare
from ..compilers.cuda import CudaCompiler
from ..interpreter.type_checking import NoneType
from . import NewExtensionModule, ModuleInfo
from ..interpreterbase import (
def _validate_nvcc_arch_args(self, args: T.Tuple[T.Union[str, CudaCompiler], T.List[str]], kwargs: ArchFlagsKwargs) -> T.Tuple[str, AutoArch, T.List[str]]:
    compiler = args[0]
    if isinstance(compiler, CudaCompiler):
        cuda_version = compiler.version
    else:
        cuda_version = compiler
    arch_list: AutoArch = args[1]
    arch_list = listify([self._break_arch_string(a) for a in arch_list])
    if len(arch_list) > 1 and (not set(arch_list).isdisjoint({'All', 'Common', 'Auto'})):
        raise InvalidArguments("The special architectures 'All', 'Common' and 'Auto' must appear alone, as a positional argument!")
    arch_list = arch_list[0] if len(arch_list) == 1 else arch_list
    detected = kwargs['detected'] if kwargs['detected'] is not None else self._detected_cc_from_compiler(compiler)
    detected = [x for a in detected for x in self._break_arch_string(a)]
    if not set(detected).isdisjoint({'All', 'Common', 'Auto'}):
        raise InvalidArguments("The special architectures 'All', 'Common' and 'Auto' must appear alone, as a positional argument!")
    return (cuda_version, arch_list, detected)