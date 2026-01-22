from __future__ import annotations
import os
import re
import subprocess
import typing as T
from .. import mlog
from .. import mesonlib
from ..compilers.compilers import CrossNoRunException
from ..mesonlib import (
from ..environment import detect_cpu_family
from .base import DependencyException, DependencyMethods, DependencyTypeName, SystemDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
class GLDependencySystem(SystemDependency):

    def __init__(self, name: str, environment: 'Environment', kwargs: T.Dict[str, T.Any]) -> None:
        super().__init__(name, environment, kwargs)
        if self.env.machines[self.for_machine].is_darwin():
            self.is_found = True
            self.link_args = ['-framework', 'OpenGL']
            return
        elif self.env.machines[self.for_machine].is_windows():
            self.is_found = True
            self.link_args = ['-lopengl32']
            return
        else:
            links = self.clib_compiler.find_library('GL', environment, [])
            has_header = self.clib_compiler.has_header('GL/gl.h', '', environment)[0]
            if links and has_header:
                self.is_found = True
                self.link_args = links
            elif links:
                raise DependencyException('Found GL runtime library but no development header files')