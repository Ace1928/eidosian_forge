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
def _detect_dep(self, name: str, package_version: str, modules: T.List[T.Tuple[str, bool]], components: T.List[T.Tuple[str, bool]], args: T.List[str]) -> None:
    mlog.debug('\nDetermining dependency {!r} with CMake executable {!r}'.format(name, self.cmakebin.executable_path()))
    gen_list = []
    if CMakeDependency.class_working_generator is not None:
        gen_list += [CMakeDependency.class_working_generator]
    gen_list += CMakeDependency.class_cmake_generators
    comp_mapped = self._map_component_list(modules, components)
    toolchain = CMakeToolchain(self.cmakebin, self.env, self.for_machine, CMakeExecScope.DEPENDENCY, self._get_build_dir())
    toolchain.write()
    for i in gen_list:
        mlog.debug('Try CMake generator: {}'.format(i if len(i) > 0 else 'auto'))
        cmake_opts = []
        cmake_opts += [f'-DNAME={name}']
        cmake_opts += ['-DARCHS={}'.format(';'.join(self.cmakeinfo.archs))]
        cmake_opts += [f'-DVERSION={package_version}']
        cmake_opts += ['-DCOMPS={}'.format(';'.join([x[0] for x in comp_mapped]))]
        cmake_opts += ['-DSTATIC={}'.format('ON' if self.static else 'OFF')]
        cmake_opts += args
        cmake_opts += self.traceparser.trace_args()
        cmake_opts += toolchain.get_cmake_args()
        cmake_opts += self._extra_cmake_opts()
        cmake_opts += ['.']
        if len(i) > 0:
            cmake_opts = ['-G', i] + cmake_opts
        ret1, out1, err1 = self._call_cmake(cmake_opts, self._main_cmake_file())
        if ret1 == 0:
            CMakeDependency.class_working_generator = i
            break
        mlog.debug(f'CMake failed for generator {i} and package {name} with error code {ret1}')
        mlog.debug(f'OUT:\n{out1}\n\n\nERR:\n{err1}\n\n')
    if ret1 != 0:
        return
    try:
        self.traceparser.parse(err1)
    except CMakeException as e:
        e2 = self._gen_exception(str(e))
        if self.required:
            raise
        else:
            self.compile_args = []
            self.link_args = []
            self.is_found = False
            self.reason = e2
            return
    self.is_found = self.traceparser.var_to_bool('PACKAGE_FOUND')
    if not self.is_found:
        return
    vers_raw = self.traceparser.get_cmake_var('PACKAGE_VERSION')
    if len(vers_raw) > 0:
        self.version = vers_raw[0]
        self.version.strip('"\' ')
    modules = self._map_module_list(modules, components)
    autodetected_module_list = False
    if len(modules) == 0:
        for i in self.traceparser.targets:
            tg = i.lower()
            lname = name.lower()
            if f'{lname}::{lname}' == tg or lname == tg.replace('::', ''):
                mlog.debug(f"Guessed CMake target '{i}'")
                modules = [(i, True)]
                autodetected_module_list = True
                break
    if len(modules) == 0:
        partial_modules: T.List[CMakeTarget] = []
        for k, v in self.traceparser.targets.items():
            tg = k.lower()
            lname = name.lower()
            if tg.startswith(f'{lname}::'):
                partial_modules += [v]
        if partial_modules:
            mlog.warning(textwrap.dedent(f"                    Could not find and exact match for the CMake dependency {name}.\n\n                    However, Meson found the following partial matches:\n\n                        {[x.name for x in partial_modules]}\n\n                    Using imported is recommended, since this approach is less error prone\n                    and better supported by Meson. Consider explicitly specifying one of\n                    these in the dependency call with:\n\n                        dependency('{name}', modules: ['{name}::<name>', ...])\n\n                    Meson will now continue to use the old-style {name}_LIBRARIES CMake\n                    variables to extract the dependency information since no explicit\n                    target is currently specified.\n\n                "))
            mlog.debug('More info for the partial match targets:')
            for tgt in partial_modules:
                mlog.debug(tgt)
        incDirs = [x for x in self.traceparser.get_cmake_var('PACKAGE_INCLUDE_DIRS') if x]
        defs = [x for x in self.traceparser.get_cmake_var('PACKAGE_DEFINITIONS') if x]
        libs_raw = [x for x in self.traceparser.get_cmake_var('PACKAGE_LIBRARIES') if x]
        libs: T.List[str] = []
        cfg_matches = True
        is_debug = cmake_is_debug(self.env)
        cm_tag_map = {'debug': is_debug, 'optimized': not is_debug, 'general': True}
        for i in libs_raw:
            if i.lower() in cm_tag_map:
                cfg_matches = cm_tag_map[i.lower()]
                continue
            if cfg_matches:
                libs += [i]
            cfg_matches = True
        if len(libs) > 0:
            self.compile_args = [f'-I{x}' for x in incDirs] + defs
            self.link_args = []
            for j in libs:
                rtgt = resolve_cmake_trace_targets(j, self.traceparser, self.env, clib_compiler=self.clib_compiler)
                self.link_args += rtgt.libraries
                self.compile_args += [f'-I{x}' for x in rtgt.include_directories]
                self.compile_args += rtgt.public_compile_opts
            mlog.debug(f'using old-style CMake variables for dependency {name}')
            mlog.debug(f'Include Dirs:         {incDirs}')
            mlog.debug(f'Compiler Definitions: {defs}')
            mlog.debug(f'Libraries:            {libs}')
            return
        self.is_found = False
        raise self._gen_exception('CMake: failed to guess a CMake target for {}.\nTry to explicitly specify one or more targets with the "modules" property.\nValid targets are:\n{}'.format(name, list(self.traceparser.targets.keys())))
    incDirs = []
    compileOptions = []
    libraries = []
    for i, required in modules:
        if i not in self.traceparser.targets:
            if not required:
                mlog.warning('CMake: T.Optional module', mlog.bold(self._original_module_name(i)), 'for', mlog.bold(name), 'was not found')
                continue
            raise self._gen_exception('CMake: invalid module {} for {}.\nTry to explicitly specify one or more targets with the "modules" property.\nValid targets are:\n{}'.format(self._original_module_name(i), name, list(self.traceparser.targets.keys())))
        if not autodetected_module_list:
            self.found_modules += [i]
        rtgt = resolve_cmake_trace_targets(i, self.traceparser, self.env, clib_compiler=self.clib_compiler, not_found_warning=lambda x: mlog.warning('CMake: Dependency', mlog.bold(x), 'for', mlog.bold(name), 'was not found'))
        incDirs += rtgt.include_directories
        compileOptions += rtgt.public_compile_opts
        libraries += rtgt.libraries + rtgt.link_flags
    incDirs = sorted(set(incDirs))
    compileOptions = sorted(set(compileOptions))
    libraries = sorted(set(libraries))
    mlog.debug(f'Include Dirs:         {incDirs}')
    mlog.debug(f'Compiler Options:     {compileOptions}')
    mlog.debug(f'Libraries:            {libraries}')
    self.compile_args = compileOptions + [f'-I{x}' for x in incDirs]
    self.link_args = libraries