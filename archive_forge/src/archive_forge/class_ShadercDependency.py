from __future__ import annotations
import functools
import re
import typing as T
from .. import mesonlib
from .. import mlog
from .base import DependencyException, DependencyMethods
from .base import BuiltinDependency, SystemDependency
from .cmake import CMakeDependency, CMakeDependencyFactory
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory, factory_methods
from .pkgconfig import PkgConfigDependency
class ShadercDependency(SystemDependency):

    def __init__(self, environment: 'Environment', kwargs: T.Dict[str, T.Any]):
        super().__init__('shaderc', environment, kwargs)
        static_lib = 'shaderc_combined'
        shared_lib = 'shaderc_shared'
        libs = [shared_lib, static_lib]
        if self.static:
            libs.reverse()
        cc = self.get_compiler()
        for lib in libs:
            self.link_args = cc.find_library(lib, environment, [])
            if self.link_args is not None:
                self.is_found = True
                if self.static and lib != static_lib:
                    mlog.warning(f'Static library {static_lib!r} not found for dependency {self.name!r}, may not be statically linked')
                break