from __future__ import annotations
import os
import errno
import shutil
import subprocess
import sys
from pathlib import Path
from ._backend import Backend
from string import Template
from itertools import chain
import warnings
class MesonTemplate:
    """Template meson build file generation class."""

    def __init__(self, modulename: str, sources: list[Path], deps: list[str], libraries: list[str], library_dirs: list[Path], include_dirs: list[Path], object_files: list[Path], linker_args: list[str], c_args: list[str], build_type: str, python_exe: str):
        self.modulename = modulename
        self.build_template_path = Path(__file__).parent.absolute() / 'meson.build.template'
        self.sources = sources
        self.deps = deps
        self.libraries = libraries
        self.library_dirs = library_dirs
        if include_dirs is not None:
            self.include_dirs = include_dirs
        else:
            self.include_dirs = []
        self.substitutions = {}
        self.objects = object_files
        self.pipeline = [self.initialize_template, self.sources_substitution, self.deps_substitution, self.include_substitution, self.libraries_substitution]
        self.build_type = build_type
        self.python_exe = python_exe

    def meson_build_template(self) -> str:
        if not self.build_template_path.is_file():
            raise FileNotFoundError(errno.ENOENT, f'Meson build template {self.build_template_path.absolute()} does not exist.')
        return self.build_template_path.read_text()

    def initialize_template(self) -> None:
        self.substitutions['modulename'] = self.modulename
        self.substitutions['buildtype'] = self.build_type
        self.substitutions['python'] = self.python_exe

    def sources_substitution(self) -> None:
        indent = ' ' * 21
        self.substitutions['source_list'] = f',\n{indent}'.join([f"{indent}'{source}'" for source in self.sources])

    def deps_substitution(self) -> None:
        indent = ' ' * 21
        self.substitutions['dep_list'] = f',\n{indent}'.join([f"{indent}dependency('{dep}')" for dep in self.deps])

    def libraries_substitution(self) -> None:
        self.substitutions['lib_dir_declarations'] = '\n'.join([f"lib_dir_{i} = declare_dependency(link_args : ['-L{lib_dir}'])" for i, lib_dir in enumerate(self.library_dirs)])
        self.substitutions['lib_declarations'] = '\n'.join([f"{lib} = declare_dependency(link_args : ['-l{lib}'])" for lib in self.libraries])
        indent = ' ' * 21
        self.substitutions['lib_list'] = f'\n{indent}'.join([f'{indent}{lib},' for lib in self.libraries])
        self.substitutions['lib_dir_list'] = f'\n{indent}'.join([f'{indent}lib_dir_{i},' for i in range(len(self.library_dirs))])

    def include_substitution(self) -> None:
        indent = ' ' * 21
        self.substitutions['inc_list'] = f',\n{indent}'.join([f"{indent}'{inc}'" for inc in self.include_dirs])

    def generate_meson_build(self):
        for node in self.pipeline:
            node()
        template = Template(self.meson_build_template())
        return template.substitute(self.substitutions)