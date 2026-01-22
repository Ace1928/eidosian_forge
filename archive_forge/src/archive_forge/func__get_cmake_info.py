from __future__ import annotations
from .base import ExternalDependency, DependencyException, DependencyTypeName
from ..mesonlib import is_windows, MesonException, PerMachine, stringlistify, extract_as_list
from ..cmake import CMakeExecutor, CMakeTraceParser, CMakeException, CMakeToolchain, CMakeExecScope, check_cmake_args, resolve_cmake_trace_targets, cmake_is_debug
from .. import mlog
import importlib.resources
from pathlib import Path
import functools
import re
import os
import shutil
import textwrap
import typing as T
def _get_cmake_info(self, cm_args: T.List[str]) -> T.Optional[CMakeInfo]:
    mlog.debug('Extracting basic cmake information')
    gen_list = []
    if CMakeDependency.class_working_generator is not None:
        gen_list += [CMakeDependency.class_working_generator]
    gen_list += CMakeDependency.class_cmake_generators
    temp_parser = CMakeTraceParser(self.cmakebin.version(), self._get_build_dir(), self.env)
    toolchain = CMakeToolchain(self.cmakebin, self.env, self.for_machine, CMakeExecScope.DEPENDENCY, self._get_build_dir())
    toolchain.write()
    for i in gen_list:
        mlog.debug('Try CMake generator: {}'.format(i if len(i) > 0 else 'auto'))
        cmake_opts = temp_parser.trace_args() + toolchain.get_cmake_args() + ['.']
        cmake_opts += cm_args
        if len(i) > 0:
            cmake_opts = ['-G', i] + cmake_opts
        ret1, out1, err1 = self._call_cmake(cmake_opts, 'CMakePathInfo.txt')
        if ret1 == 0:
            CMakeDependency.class_working_generator = i
            break
        mlog.debug(f'CMake failed to gather system information for generator {i} with error code {ret1}')
        mlog.debug(f'OUT:\n{out1}\n\n\nERR:\n{err1}\n\n')
    if ret1 != 0:
        return None
    try:
        temp_parser.parse(err1)
    except MesonException:
        return None

    def process_paths(l: T.List[str]) -> T.Set[str]:
        if is_windows():
            tmp = [x.split(os.pathsep) for x in l]
        else:
            tmp = [re.split(':|;', x) for x in l]
        flattened = [x for sublist in tmp for x in sublist]
        return set(flattened)
    root_paths_set = process_paths(temp_parser.get_cmake_var('MESON_FIND_ROOT_PATH'))
    root_paths_set.update(process_paths(temp_parser.get_cmake_var('MESON_CMAKE_SYSROOT')))
    root_paths = sorted(root_paths_set)
    root_paths = [x for x in root_paths if os.path.isdir(x)]
    module_paths_set = process_paths(temp_parser.get_cmake_var('MESON_PATHS_LIST'))
    rooted_paths: T.List[str] = []
    for j in [Path(x) for x in root_paths]:
        for p in [Path(x) for x in module_paths_set]:
            rooted_paths.append(str(j / p.relative_to(p.anchor)))
    module_paths = sorted(module_paths_set.union(rooted_paths))
    module_paths = [x for x in module_paths if os.path.isdir(x)]
    archs = temp_parser.get_cmake_var('MESON_ARCH_LIST')
    common_paths = ['lib', 'lib32', 'lib64', 'libx32', 'share', '']
    for i in archs:
        common_paths += [os.path.join('lib', i)]
    res = CMakeInfo(module_paths=module_paths, cmake_root=temp_parser.get_cmake_var('MESON_CMAKE_ROOT')[0], archs=archs, common_paths=common_paths)
    mlog.debug(f'  -- Module search paths:    {res.module_paths}')
    mlog.debug(f'  -- CMake root:             {res.cmake_root}')
    mlog.debug(f'  -- CMake architectures:    {res.archs}')
    mlog.debug(f'  -- CMake lib search paths: {res.common_paths}')
    return res