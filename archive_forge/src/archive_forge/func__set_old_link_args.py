from __future__ import annotations
import glob
import os
import re
import pathlib
import shutil
import subprocess
import typing as T
import functools
from mesonbuild.interpreterbase.decorators import FeatureDeprecated
from .. import mesonlib, mlog
from ..environment import get_llvm_tool_names
from ..mesonlib import version_compare, version_compare_many, search_version, stringlistify, extract_as_list
from .base import DependencyException, DependencyMethods, detect_compiler, strip_system_includedirs, strip_system_libdirs, SystemDependency, ExternalDependency, DependencyTypeName
from .cmake import CMakeDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
from .misc import threads_factory
from .pkgconfig import PkgConfigDependency
def _set_old_link_args(self) -> None:
    """Setting linker args for older versions of llvm.

        Old versions of LLVM bring an extra level of insanity with them.
        llvm-config will provide the correct arguments for static linking, but
        not for shared-linking, we have to figure those out ourselves, because
        of course we do.
        """
    if self.static:
        self.link_args = self.get_config_value(['--libs', '--ldflags', '--system-libs'] + list(self.required_modules), 'link_args')
    else:
        libdir = self.get_config_value(['--libdir'], 'link_args')[0]
        expected_name = f'libLLVM-{self.version}'
        re_name = re.compile(f'{expected_name}.(so|dll|dylib)$')
        for file_ in os.listdir(libdir):
            if re_name.match(file_):
                self.link_args = [f'-L{libdir}', '-l{}'.format(os.path.splitext(file_.lstrip('lib'))[0])]
                break
        else:
            raise DependencyException('Could not find a dynamically linkable library for LLVM.')