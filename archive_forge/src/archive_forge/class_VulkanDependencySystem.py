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
class VulkanDependencySystem(SystemDependency):

    def __init__(self, name: str, environment: 'Environment', kwargs: T.Dict[str, T.Any], language: T.Optional[str]=None) -> None:
        super().__init__(name, environment, kwargs, language=language)
        try:
            self.vulkan_sdk = os.environ['VULKAN_SDK']
            if not os.path.isabs(self.vulkan_sdk):
                raise DependencyException('VULKAN_SDK must be an absolute path.')
        except KeyError:
            self.vulkan_sdk = None
        if self.vulkan_sdk:
            lib_name = 'vulkan'
            lib_dir = 'lib'
            inc_dir = 'include'
            if mesonlib.is_windows():
                lib_name = 'vulkan-1'
                lib_dir = 'Lib32'
                inc_dir = 'Include'
                if detect_cpu_family(self.env.coredata.compilers.host) == 'x86_64':
                    lib_dir = 'Lib'
            inc_path = os.path.join(self.vulkan_sdk, inc_dir)
            header = os.path.join(inc_path, 'vulkan', 'vulkan.h')
            lib_path = os.path.join(self.vulkan_sdk, lib_dir)
            find_lib = self.clib_compiler.find_library(lib_name, environment, [lib_path])
            if not find_lib:
                raise DependencyException('VULKAN_SDK point to invalid directory (no lib)')
            if not os.path.isfile(header):
                raise DependencyException('VULKAN_SDK point to invalid directory (no include)')
            self.type_name = DependencyTypeName('vulkan_sdk')
            self.is_found = True
            self.compile_args.append('-I' + inc_path)
            self.link_args.append('-L' + lib_path)
            self.link_args.append('-l' + lib_name)
        else:
            libs = self.clib_compiler.find_library('vulkan', environment, [])
            if libs is not None and self.clib_compiler.has_header('vulkan/vulkan.h', '', environment, disable_cache=True)[0]:
                self.is_found = True
                for lib in libs:
                    self.link_args.append(lib)
        if self.is_found:
            get_version = '#include <stdio.h>\n#include <vulkan/vulkan.h>\n\nint main() {\n    printf("%i.%i.%i", VK_VERSION_MAJOR(VK_HEADER_VERSION_COMPLETE),\n                       VK_VERSION_MINOR(VK_HEADER_VERSION_COMPLETE),\n                       VK_VERSION_PATCH(VK_HEADER_VERSION_COMPLETE));\n    return 0;\n}\n'
            try:
                run = self.clib_compiler.run(get_version, environment, extra_args=self.compile_args)
            except CrossNoRunException:
                run = None
            if run and run.compiled and (run.returncode == 0):
                self.version = run.stdout
            elif self.vulkan_sdk:
                match = re.search(f'VulkanSDK{re.escape(os.path.sep)}([0-9]+(?:\\.[0-9]+)+)', self.vulkan_sdk)
                if match:
                    self.version = match.group(1)
                else:
                    mlog.warning(f'Environment variable VULKAN_SDK={self.vulkan_sdk} is present, but Vulkan version could not be extracted.')