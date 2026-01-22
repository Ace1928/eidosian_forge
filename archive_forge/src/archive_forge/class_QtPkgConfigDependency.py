from __future__ import annotations
import abc
import re
import os
import typing as T
from .base import DependencyException, DependencyMethods
from .configtool import ConfigToolDependency
from .detect import packages
from .framework import ExtraFrameworkDependency
from .pkgconfig import PkgConfigDependency
from .factory import DependencyFactory
from .. import mlog
from .. import mesonlib
class QtPkgConfigDependency(_QtBase, PkgConfigDependency, metaclass=abc.ABCMeta):
    """Specialization of the PkgConfigDependency for Qt."""

    def __init__(self, name: str, env: 'Environment', kwargs: T.Dict[str, T.Any]):
        _QtBase.__init__(self, name, kwargs)
        PkgConfigDependency.__init__(self, self.qtpkgname + 'Core', env, kwargs)
        if 'Core' not in self.requested_modules:
            self.compile_args = []
            self.link_args = []
        for m in self.requested_modules:
            mod = PkgConfigDependency(self.qtpkgname + m, self.env, kwargs, language=self.language)
            if not mod.found():
                self.is_found = False
                return
            if self.private_headers:
                qt_inc_dir = mod.get_variable(pkgconfig='includedir')
                mod_private_dir = os.path.join(qt_inc_dir, 'Qt' + m)
                if not os.path.isdir(mod_private_dir):
                    mod_private_dir = qt_inc_dir
                mod_private_inc = _qt_get_private_includes(mod_private_dir, m, mod.version)
                for directory in mod_private_inc:
                    mod.compile_args.append('-I' + directory)
            self._add_sub_dependency([lambda: mod])
        if self.env.machines[self.for_machine].is_windows() and self.qtmain:
            debug_lib_name = self.qtpkgname + 'Core' + _get_modules_lib_suffix(self.version, self.env.machines[self.for_machine], True)
            is_debug = False
            for arg in self.get_link_args():
                if arg == f'-l{debug_lib_name}' or arg.endswith(f'{debug_lib_name}.lib') or arg.endswith(f'{debug_lib_name}.a'):
                    is_debug = True
                    break
            libdir = self.get_variable(pkgconfig='libdir')
            if not self._link_with_qt_winmain(is_debug, libdir):
                self.is_found = False
                return
        self.bindir = self.get_pkgconfig_host_bins(self)
        if not self.bindir:
            prefix = self.get_variable(pkgconfig='exec_prefix')
            if prefix:
                self.bindir = os.path.join(prefix, 'bin')
        self.libexecdir = self.get_pkgconfig_host_libexecs(self)

    @staticmethod
    @abc.abstractmethod
    def get_pkgconfig_host_bins(core: PkgConfigDependency) -> T.Optional[str]:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_pkgconfig_host_libexecs(core: PkgConfigDependency) -> T.Optional[str]:
        pass

    @abc.abstractmethod
    def get_private_includes(self, mod_inc_dir: str, module: str) -> T.List[str]:
        pass

    def log_info(self) -> str:
        return 'pkg-config'