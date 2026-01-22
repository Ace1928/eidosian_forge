import argparse
import functools
import itertools
import marshal
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List
class Freezer:

    def __init__(self, verbose: bool):
        self.frozen_modules: List[FrozenModule] = []
        self.indent: int = 0
        self.verbose: bool = verbose

    def msg(self, path: Path, code: str):
        if not self.verbose:
            return
        for i in range(self.indent):
            print('    ', end='')
        print(f'{code} {path}')

    def write_bytecode(self, install_root):
        """
        Write the `.c` files containing the frozen bytecode.

        Shared frozen modules evenly across the files.
        """
        bytecode_file_names = [f'bytecode_{i}.c' for i in range(NUM_BYTECODE_FILES)]
        bytecode_files = [open(os.path.join(install_root, name), 'w') for name in bytecode_file_names]
        it = itertools.cycle(bytecode_files)
        for m in self.frozen_modules:
            self.write_frozen(m, next(it))
        for f in bytecode_files:
            f.close()

    def write_main(self, install_root, oss, symbol_name):
        """Write the `main.c` file containing a table enumerating all the frozen modules."""
        with open(os.path.join(install_root, 'main.c'), 'w') as outfp:
            outfp.write(MAIN_INCLUDES)
            for m in self.frozen_modules:
                outfp.write(f'extern unsigned char {m.c_name}[];\n')
            outfp.write(MAIN_PREFIX_TEMPLATE.format(symbol_name))
            for m in self.frozen_modules:
                outfp.write(f'\t{{"{m.module_name}", {m.c_name}, {m.size}}},\n')
            outfp.write(MAIN_SUFFIX)
            if oss:
                outfp.write(FAKE_PREFIX)
                outfp.write(MAIN_SUFFIX)

    def write_frozen(self, m: FrozenModule, outfp):
        """Write a single frozen module's bytecode out to a C variable."""
        outfp.write(f'unsigned char {m.c_name}[] = {{')
        for i in range(0, len(m.bytecode), 16):
            outfp.write('\n\t')
            for c in bytes(m.bytecode[i:i + 16]):
                outfp.write('%d,' % c)
        outfp.write('\n};\n')

    def compile_path(self, path: Path, top_package_path: Path):
        """Entry point for compiling a Path object."""
        if path.is_dir():
            self.compile_package(path, top_package_path)
        else:
            self.compile_file(path, top_package_path)

    @indent_msg
    def compile_package(self, path: Path, top_package_path: Path):
        """Compile all the files within a Python package dir."""
        assert path.is_dir()
        if path.name in DENY_LIST:
            self.msg(path, 'X')
            return
        is_package_dir = any((child.name == '__init__.py' for child in path.iterdir()))
        if not is_package_dir:
            self.msg(path, 'S')
            return
        self.msg(path, 'P')
        for child in path.iterdir():
            self.compile_path(child, top_package_path)

    def get_module_qualname(self, file_path: Path, top_package_path: Path) -> List[str]:
        normalized_path = file_path.relative_to(top_package_path.parent)
        if normalized_path.name == '__init__.py':
            module_basename = normalized_path.parent.name
            module_parent = normalized_path.parent.parent.parts
        else:
            module_basename = normalized_path.stem
            module_parent = normalized_path.parent.parts
        return list(module_parent) + [module_basename]

    def compile_string(self, file_content: str) -> types.CodeType:
        path_marker = PATH_MARKER
        return compile(file_content, path_marker, 'exec')

    @indent_msg
    def compile_file(self, path: Path, top_package_path: Path):
        """
        Compile a Python source file to frozen bytecode.

        Append the result to `self.frozen_modules`.
        """
        assert path.is_file()
        if path.suffix != '.py':
            self.msg(path, 'N')
            return
        if path.name in DENY_LIST:
            self.msg(path, 'X')
            return
        self.msg(path, 'F')
        module_qualname = self.get_module_qualname(path, top_package_path)
        module_mangled_name = '__'.join(module_qualname)
        c_name = 'M_' + module_mangled_name
        with open(path) as src_file:
            co = self.compile_string(src_file.read())
        bytecode = marshal.dumps(co)
        size = len(bytecode)
        if path.name == '__init__.py':
            size = -size
        self.frozen_modules.append(FrozenModule('.'.join(module_qualname), c_name, size, bytecode))