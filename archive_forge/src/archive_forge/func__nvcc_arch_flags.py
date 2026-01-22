from __future__ import annotations
import re
import typing as T
from ..mesonlib import listify, version_compare
from ..compilers.cuda import CudaCompiler
from ..interpreter.type_checking import NoneType
from . import NewExtensionModule, ModuleInfo
from ..interpreterbase import (
def _nvcc_arch_flags(self, cuda_version: str, cuda_arch_list: AutoArch, detected: T.List[str]) -> T.Tuple[T.List[str], T.List[str]]:
    """
        Using the CUDA Toolkit version and the target architectures, compute
        the NVCC architecture flags.
        """
    cuda_known_gpu_architectures = ['Fermi', 'Kepler', 'Maxwell']
    cuda_common_gpu_architectures = ['3.0', '3.5', '5.0']
    cuda_hi_limit_gpu_architecture = None
    cuda_lo_limit_gpu_architecture = '2.0'
    cuda_all_gpu_architectures = ['3.0', '3.2', '3.5', '5.0']
    if version_compare(cuda_version, '<7.0'):
        cuda_hi_limit_gpu_architecture = '5.2'
    if version_compare(cuda_version, '>=7.0'):
        cuda_known_gpu_architectures += ['Kepler+Tegra', 'Kepler+Tesla', 'Maxwell+Tegra']
        cuda_common_gpu_architectures += ['5.2']
        if version_compare(cuda_version, '<8.0'):
            cuda_common_gpu_architectures += ['5.2+PTX']
            cuda_hi_limit_gpu_architecture = '6.0'
    if version_compare(cuda_version, '>=8.0'):
        cuda_known_gpu_architectures += ['Pascal', 'Pascal+Tegra']
        cuda_common_gpu_architectures += ['6.0', '6.1']
        cuda_all_gpu_architectures += ['6.0', '6.1', '6.2']
        if version_compare(cuda_version, '<9.0'):
            cuda_common_gpu_architectures += ['6.1+PTX']
            cuda_hi_limit_gpu_architecture = '7.0'
    if version_compare(cuda_version, '>=9.0'):
        cuda_known_gpu_architectures += ['Volta', 'Xavier']
        cuda_common_gpu_architectures += ['7.0']
        cuda_all_gpu_architectures += ['7.0', '7.2']
        cuda_lo_limit_gpu_architecture = '3.0'
        if version_compare(cuda_version, '<10.0'):
            cuda_common_gpu_architectures += ['7.2+PTX']
            cuda_hi_limit_gpu_architecture = '8.0'
    if version_compare(cuda_version, '>=10.0'):
        cuda_known_gpu_architectures += ['Turing']
        cuda_common_gpu_architectures += ['7.5']
        cuda_all_gpu_architectures += ['7.5']
        if version_compare(cuda_version, '<11.0'):
            cuda_common_gpu_architectures += ['7.5+PTX']
            cuda_hi_limit_gpu_architecture = '8.0'
    cuda_ampere_bin = ['8.0']
    cuda_ampere_ptx = ['8.0']
    if version_compare(cuda_version, '>=11.0'):
        cuda_known_gpu_architectures += ['Ampere']
        cuda_common_gpu_architectures += ['8.0']
        cuda_all_gpu_architectures += ['8.0']
        cuda_lo_limit_gpu_architecture = '3.5'
        if version_compare(cuda_version, '<11.1'):
            cuda_common_gpu_architectures += ['8.0+PTX']
            cuda_hi_limit_gpu_architecture = '8.6'
    if version_compare(cuda_version, '>=11.1'):
        cuda_ampere_bin += ['8.6']
        cuda_ampere_ptx = ['8.6']
        cuda_common_gpu_architectures += ['8.6']
        cuda_all_gpu_architectures += ['8.6']
        if version_compare(cuda_version, '<11.8'):
            cuda_common_gpu_architectures += ['8.6+PTX']
            cuda_hi_limit_gpu_architecture = '8.7'
    if version_compare(cuda_version, '>=11.8'):
        cuda_known_gpu_architectures += ['Orin', 'Lovelace', 'Hopper']
        cuda_common_gpu_architectures += ['8.9', '9.0', '9.0+PTX']
        cuda_all_gpu_architectures += ['8.7', '8.9', '9.0']
        if version_compare(cuda_version, '<12'):
            cuda_hi_limit_gpu_architecture = '9.1'
    if version_compare(cuda_version, '>=12.0'):
        cuda_lo_limit_gpu_architecture = '5.0'
        if version_compare(cuda_version, '<13'):
            cuda_hi_limit_gpu_architecture = '10.0'
    if not cuda_arch_list:
        cuda_arch_list = 'Auto'
    if cuda_arch_list == 'All':
        cuda_arch_list = cuda_known_gpu_architectures
    elif cuda_arch_list == 'Common':
        cuda_arch_list = cuda_common_gpu_architectures
    elif cuda_arch_list == 'Auto':
        if detected:
            if isinstance(detected, list):
                cuda_arch_list = detected
            else:
                cuda_arch_list = self._break_arch_string(detected)
            cuda_arch_list = self._filter_cuda_arch_list(cuda_arch_list, cuda_lo_limit_gpu_architecture, cuda_hi_limit_gpu_architecture, cuda_common_gpu_architectures[-1])
        else:
            cuda_arch_list = cuda_common_gpu_architectures
    elif isinstance(cuda_arch_list, str):
        cuda_arch_list = self._break_arch_string(cuda_arch_list)
    cuda_arch_list = sorted((x for x in set(cuda_arch_list) if x))
    cuda_arch_bin: T.List[str] = []
    cuda_arch_ptx: T.List[str] = []
    for arch_name in cuda_arch_list:
        arch_bin: T.Optional[T.List[str]]
        arch_ptx: T.Optional[T.List[str]]
        add_ptx = arch_name.endswith('+PTX')
        if add_ptx:
            arch_name = arch_name[:-len('+PTX')]
        if re.fullmatch('[0-9]+\\.[0-9](\\([0-9]+\\.[0-9]\\))?', arch_name):
            arch_bin, arch_ptx = ([arch_name], [arch_name])
        else:
            arch_bin, arch_ptx = {'Fermi': (['2.0', '2.1(2.0)'], []), 'Kepler+Tegra': (['3.2'], []), 'Kepler+Tesla': (['3.7'], []), 'Kepler': (['3.0', '3.5'], ['3.5']), 'Maxwell+Tegra': (['5.3'], []), 'Maxwell': (['5.0', '5.2'], ['5.2']), 'Pascal': (['6.0', '6.1'], ['6.1']), 'Pascal+Tegra': (['6.2'], []), 'Volta': (['7.0'], ['7.0']), 'Xavier': (['7.2'], []), 'Turing': (['7.5'], ['7.5']), 'Ampere': (cuda_ampere_bin, cuda_ampere_ptx), 'Orin': (['8.7'], []), 'Lovelace': (['8.9'], ['8.9']), 'Hopper': (['9.0'], ['9.0'])}.get(arch_name, (None, None))
        if arch_bin is None:
            raise InvalidArguments(f'Unknown CUDA Architecture Name {arch_name}!')
        cuda_arch_bin += arch_bin
        if add_ptx:
            if not arch_ptx:
                arch_ptx = arch_bin
            cuda_arch_ptx += arch_ptx
    cuda_arch_bin = sorted(set(cuda_arch_bin))
    cuda_arch_ptx = sorted(set(cuda_arch_ptx))
    nvcc_flags = []
    nvcc_archs_readable = []
    for arch in cuda_arch_bin:
        arch, codev = re.fullmatch('([0-9]+\\.[0-9])(?:\\(([0-9]+\\.[0-9])\\))?', arch).groups()
        if version_compare(arch, '<' + cuda_lo_limit_gpu_architecture):
            continue
        if cuda_hi_limit_gpu_architecture and version_compare(arch, '>=' + cuda_hi_limit_gpu_architecture):
            continue
        if codev:
            arch = arch.replace('.', '')
            codev = codev.replace('.', '')
            nvcc_flags += ['-gencode', 'arch=compute_' + codev + ',code=sm_' + arch]
            nvcc_archs_readable += ['sm_' + arch]
        else:
            arch = arch.replace('.', '')
            nvcc_flags += ['-gencode', 'arch=compute_' + arch + ',code=sm_' + arch]
            nvcc_archs_readable += ['sm_' + arch]
    for arch in cuda_arch_ptx:
        arch, codev = re.fullmatch('([0-9]+\\.[0-9])(?:\\(([0-9]+\\.[0-9])\\))?', arch).groups()
        if codev:
            arch = codev
        if version_compare(arch, '<' + cuda_lo_limit_gpu_architecture):
            continue
        if cuda_hi_limit_gpu_architecture and version_compare(arch, '>=' + cuda_hi_limit_gpu_architecture):
            continue
        arch = arch.replace('.', '')
        nvcc_flags += ['-gencode', 'arch=compute_' + arch + ',code=compute_' + arch]
        nvcc_archs_readable += ['compute_' + arch]
    return (nvcc_flags, nvcc_archs_readable)