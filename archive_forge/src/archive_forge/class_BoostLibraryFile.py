from __future__ import annotations
import re
import dataclasses
import functools
import typing as T
from pathlib import Path
from .. import mlog
from .. import mesonlib
from .base import DependencyException, SystemDependency
from .detect import packages
from .pkgconfig import PkgConfigDependency
from .misc import threads_factory
@functools.total_ordering
class BoostLibraryFile:
    boost_python_libs = ['boost_python', 'boost_numpy']
    reg_python_mod_split = re.compile('(boost_[a-zA-Z]+)([0-9]*)')
    reg_abi_tag = re.compile('^s?g?y?d?p?n?$')
    reg_ver_tag = re.compile('^[0-9_]+$')

    def __init__(self, path: Path):
        self.path = path
        self.name = self.path.name
        self.static = False
        self.toolset = ''
        self.arch = ''
        self.version_lib = ''
        self.mt = True
        self.runtime_static = False
        self.runtime_debug = False
        self.python_debug = False
        self.debug = False
        self.stlport = False
        self.deprecated_iostreams = False
        name_parts = self.name.split('.')
        self.basename = name_parts[0]
        self.suffixes = name_parts[1:]
        self.vers_raw = [x for x in self.suffixes if x.isdigit()]
        self.suffixes = [x for x in self.suffixes if not x.isdigit()]
        self.nvsuffix = '.'.join(self.suffixes)
        self.nametags = self.basename.split('-')
        self.mod_name = self.nametags[0]
        if self.mod_name.startswith('lib'):
            self.mod_name = self.mod_name[3:]
        if len(self.vers_raw) >= 2:
            self.version_lib = '{}_{}'.format(self.vers_raw[0], self.vers_raw[1])
        if self.nvsuffix in {'so', 'dll', 'dll.a', 'dll.lib', 'dylib'}:
            self.static = False
        elif self.nvsuffix in {'a', 'lib'}:
            self.static = True
        else:
            raise UnknownFileException(self.path)
        if self.basename.startswith('boost_') and self.nvsuffix == 'lib':
            self.static = False
        tags = self.nametags[1:]
        if self.is_python_lib():
            tags = self.fix_python_name(tags)
        if not tags:
            return
        self.mt = False
        for i in tags:
            if i == 'mt':
                self.mt = True
            elif len(i) == 3 and i[1:] in {'32', '64'}:
                self.arch = i
            elif BoostLibraryFile.reg_abi_tag.match(i):
                self.runtime_static = 's' in i
                self.runtime_debug = 'g' in i
                self.python_debug = 'y' in i
                self.debug = 'd' in i
                self.stlport = 'p' in i
                self.deprecated_iostreams = 'n' in i
            elif BoostLibraryFile.reg_ver_tag.match(i):
                self.version_lib = i
            else:
                self.toolset = i

    def __repr__(self) -> str:
        return f'<LIB: {self.abitag} {self.mod_name:<32} {self.path}>'

    def __lt__(self, other: object) -> bool:
        if isinstance(other, BoostLibraryFile):
            return (self.mod_name, self.static, self.version_lib, self.arch, not self.mt, not self.runtime_static, not self.debug, self.runtime_debug, self.python_debug, self.stlport, self.deprecated_iostreams, self.name) < (other.mod_name, other.static, other.version_lib, other.arch, not other.mt, not other.runtime_static, not other.debug, other.runtime_debug, other.python_debug, other.stlport, other.deprecated_iostreams, other.name)
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BoostLibraryFile):
            return self.name == other.name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def abitag(self) -> str:
        abitag = ''
        abitag += 'S' if self.static else '-'
        abitag += 'M' if self.mt else '-'
        abitag += ' '
        abitag += 's' if self.runtime_static else '-'
        abitag += 'g' if self.runtime_debug else '-'
        abitag += 'y' if self.python_debug else '-'
        abitag += 'd' if self.debug else '-'
        abitag += 'p' if self.stlport else '-'
        abitag += 'n' if self.deprecated_iostreams else '-'
        abitag += ' ' + (self.arch or '???')
        abitag += ' ' + (self.toolset or '?')
        abitag += ' ' + (self.version_lib or 'x_xx')
        return abitag

    def is_boost(self) -> bool:
        return any((self.name.startswith(x) for x in ['libboost_', 'boost_']))

    def is_python_lib(self) -> bool:
        return any((self.mod_name.startswith(x) for x in BoostLibraryFile.boost_python_libs))

    def fix_python_name(self, tags: T.List[str]) -> T.List[str]:
        other_tags: T.List[str] = []
        m_cur = BoostLibraryFile.reg_python_mod_split.match(self.mod_name)
        cur_name = m_cur.group(1)
        cur_vers = m_cur.group(2)

        def update_vers(new_vers: str) -> None:
            nonlocal cur_vers
            new_vers = new_vers.replace('_', '')
            new_vers = new_vers.replace('.', '')
            if not new_vers.isdigit():
                return
            if len(new_vers) > len(cur_vers):
                cur_vers = new_vers
        for i in tags:
            if i.startswith('py'):
                update_vers(i[2:])
            elif i.isdigit():
                update_vers(i)
            elif len(i) >= 3 and i[0].isdigit and i[2].isdigit() and (i[1] == '.'):
                update_vers(i)
            else:
                other_tags += [i]
        self.mod_name = cur_name + cur_vers
        return other_tags

    def mod_name_matches(self, mod_name: str) -> bool:
        if self.mod_name == mod_name:
            return True
        if not self.is_python_lib():
            return False
        m_cur = BoostLibraryFile.reg_python_mod_split.match(self.mod_name)
        m_arg = BoostLibraryFile.reg_python_mod_split.match(mod_name)
        if not m_cur or not m_arg:
            return False
        if m_cur.group(1) != m_arg.group(1):
            return False
        cur_vers = m_cur.group(2)
        arg_vers = m_arg.group(2)
        if not arg_vers:
            arg_vers = '2'
        return cur_vers.startswith(arg_vers)

    def version_matches(self, version_lib: str) -> bool:
        if not self.version_lib or not version_lib:
            return True
        return self.version_lib == version_lib

    def arch_matches(self, arch: str) -> bool:
        if not self.arch or not arch:
            return True
        return self.arch == arch

    def vscrt_matches(self, vscrt: str) -> bool:
        if not vscrt:
            return True
        if vscrt in {'/MD', '-MD'}:
            return not self.runtime_static and (not self.runtime_debug)
        elif vscrt in {'/MDd', '-MDd'}:
            return not self.runtime_static and self.runtime_debug
        elif vscrt in {'/MT', '-MT'}:
            return (self.runtime_static or not self.static) and (not self.runtime_debug)
        elif vscrt in {'/MTd', '-MTd'}:
            return (self.runtime_static or not self.static) and self.runtime_debug
        mlog.warning(f'Boost: unknown vscrt tag {vscrt}. This may cause the compilation to fail. Please consider reporting this as a bug.', once=True)
        return True

    def get_compiler_args(self) -> T.List[str]:
        args: T.List[str] = []
        if self.mod_name in boost_libraries:
            libdef = boost_libraries[self.mod_name]
            if self.static:
                args += libdef.static
            else:
                args += libdef.shared
            if self.mt:
                args += libdef.multi
            else:
                args += libdef.single
        return args

    def get_link_args(self) -> T.List[str]:
        return [self.path.as_posix()]