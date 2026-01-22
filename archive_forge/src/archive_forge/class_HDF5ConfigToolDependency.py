from __future__ import annotations
import functools
import os
import re
from pathlib import Path
from ..mesonlib import OrderedSet, join_args
from .base import DependencyException, DependencyMethods
from .configtool import ConfigToolDependency
from .detect import packages
from .pkgconfig import PkgConfigDependency, PkgConfigInterface
from .factory import factory_methods
import typing as T
class HDF5ConfigToolDependency(ConfigToolDependency):
    """Wrapper around hdf5 binary config tools."""
    version_arg = '-showconfig'

    def __init__(self, name: str, environment: 'Environment', kwargs: T.Dict[str, T.Any], language: T.Optional[str]=None) -> None:
        language = language or 'c'
        if language not in {'c', 'cpp', 'fortran'}:
            raise DependencyException(f'Language {language} is not supported with HDF5.')
        if language == 'c':
            cenv = 'CC'
            lenv = 'C'
            tools = ['h5cc', 'h5pcc']
        elif language == 'cpp':
            cenv = 'CXX'
            lenv = 'CXX'
            tools = ['h5c++', 'h5pc++']
        elif language == 'fortran':
            cenv = 'FC'
            lenv = 'F'
            tools = ['h5fc', 'h5pfc']
        else:
            raise DependencyException('How did you get here?')
        for_machine = self.get_for_machine_from_kwargs(kwargs)
        nkwargs = kwargs.copy()
        nkwargs['tools'] = tools
        compiler = environment.coredata.compilers[for_machine][language]
        try:
            os.environ[f'HDF5_{cenv}'] = join_args(compiler.get_exelist())
            os.environ[f'HDF5_{lenv}LINKER'] = join_args(compiler.get_linker_exelist())
            super().__init__(name, environment, nkwargs, language)
        finally:
            del os.environ[f'HDF5_{cenv}']
            del os.environ[f'HDF5_{lenv}LINKER']
        if not self.is_found:
            return
        args = self.get_config_value(['-show', '-c'], 'args')[1:]
        args += self.get_config_value(['-show', '-noshlib' if self.static else '-shlib'], 'args')[1:]
        found = False
        for arg in args:
            if arg.startswith(('-I', '-f', '-D')) or arg == '-pthread':
                self.compile_args.append(arg)
            elif arg.startswith(('-L', '-l', '-Wl')):
                self.link_args.append(arg)
                found = True
            elif Path(arg).is_file():
                self.link_args.append(arg)
                found = True
        if not found:
            raise DependencyException('HDF5 was built with cmake instead of autotools, and h5cc is broken.')

    def _sanitize_version(self, ver: str) -> str:
        v = re.search('\\s*HDF5 Version: (\\d+\\.\\d+\\.\\d+)', ver)
        return v.group(1)