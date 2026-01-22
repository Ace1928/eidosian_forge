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
class DlSystemDependency(SystemDependency):

    def __init__(self, name: str, env: 'Environment', kwargs: T.Dict[str, T.Any]):
        super().__init__(name, env, kwargs)
        self.feature_since = ('0.62.0', "consider checking for `dlopen` with and without `find_library('dl')`")
        h = self.clib_compiler.has_header('dlfcn.h', '', env)
        self.link_args = self.clib_compiler.find_library('dl', env, [], self.libtype)
        if h[0] and self.link_args:
            self.is_found = True