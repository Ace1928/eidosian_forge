from __future__ import annotations
import functools
import typing as T
import os
import re
from ..environment import detect_cpu_family
from .base import DependencyMethods, detect_compiler, SystemDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import factory_methods
from .pkgconfig import PkgConfigDependency
class IntelMPIConfigToolDependency(_MPIConfigToolDependency):
    """Wrapper around Intel's mpiicc and friends."""
    version_arg = '-v'

    def __init__(self, name: str, env: 'Environment', kwargs: T.Dict[str, T.Any], language: T.Optional[str]=None):
        super().__init__(name, env, kwargs, language=language)
        if not self.is_found:
            return
        args = self.get_config_value(['-show'], 'link and compile args')
        self.compile_args = self._filter_compile_args(args)
        self.link_args = self._filter_link_args(args)

    def _sanitize_version(self, out: str) -> str:
        v = re.search('(\\d{4}) Update (\\d)', out)
        if v:
            return '{}.{}'.format(v.group(1), v.group(2))
        return out