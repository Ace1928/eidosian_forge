import fnmatch
import importlib.machinery
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Generator, Sequence, Iterable, Union
from .line import (
class ModuleGatherer:

    def __init__(self, paths: Optional[Iterable[Union[str, Path]]]=None, skiplist: Optional[Sequence[str]]=None) -> None:
        """Initialize module gatherer with all modules in `paths`, which should be a list of
        directory names. If `paths` is not given, `sys.path` will be used."""
        self.modules: Set[str] = set()
        self.paths: Set[_LoadedInode] = set()
        self.skiplist: Sequence[str] = skiplist if skiplist is not None else tuple()
        self.fully_loaded = False
        if paths is None:
            self.modules.update(sys.builtin_module_names)
            paths = sys.path
        self.find_iterator = self.find_all_modules((Path(p).resolve() if p else Path.cwd() for p in paths))

    def module_matches(self, cw: str, prefix: str='') -> Set[str]:
        """Modules names to replace cw with"""
        full = f'{prefix}.{cw}' if prefix else cw
        matches = (name for name in self.modules if name.startswith(full) and name.find('.', len(full)) == -1)
        if prefix:
            return {match[len(prefix) + 1:] for match in matches}
        else:
            return set(matches)

    def attr_matches(self, cw: str, prefix: str='', only_modules: bool=False) -> Set[str]:
        """Attributes to replace name with"""
        full = f'{prefix}.{cw}' if prefix else cw
        module_name, _, name_after_dot = full.rpartition('.')
        if module_name not in sys.modules:
            return set()
        module = sys.modules[module_name]
        if only_modules:
            matches = {name for name in dir(module) if name.startswith(name_after_dot) and f'{module_name}.{name}' in sys.modules}
        else:
            matches = {name for name in dir(module) if name.startswith(name_after_dot)}
        module_part = cw.rpartition('.')[0]
        if module_part:
            matches = {f'{module_part}.{m}' for m in matches}
        return matches

    def module_attr_matches(self, name: str) -> Set[str]:
        """Only attributes which are modules to replace name with"""
        return self.attr_matches(name, only_modules=True)

    def complete(self, cursor_offset: int, line: str) -> Optional[Set[str]]:
        """Construct a full list of possibly completions for imports."""
        tokens = line.split()
        if 'from' not in tokens and 'import' not in tokens:
            return None
        result = current_word(cursor_offset, line)
        if result is None:
            return None
        from_import_from = current_from_import_from(cursor_offset, line)
        if from_import_from is not None:
            import_import = current_from_import_import(cursor_offset, line)
            if import_import is not None:
                matches = self.module_matches(import_import.word, from_import_from.word)
                matches.update(self.attr_matches(import_import.word, from_import_from.word))
            else:
                matches = self.module_attr_matches(from_import_from.word)
                matches.update(self.module_matches(from_import_from.word))
            return matches
        cur_import = current_import(cursor_offset, line)
        if cur_import is not None:
            matches = self.module_matches(cur_import.word)
            matches.update(self.module_attr_matches(cur_import.word))
            return matches
        else:
            return None

    def find_modules(self, path: Path) -> Generator[Optional[str], None, None]:
        """Find all modules (and packages) for a given directory."""
        if not path.is_dir():
            return
        if any((fnmatch.fnmatch(path.name, entry) for entry in self.skiplist)):
            return
        try:
            children = tuple(path.iterdir())
        except OSError:
            return
        finder = importlib.machinery.FileFinder(str(path), *LOADERS)
        for p in children:
            if p.name.startswith('.') or p.name == '__pycache__':
                continue
            elif any((fnmatch.fnmatch(p.name, entry) for entry in self.skiplist)):
                continue
            elif not any((p.name.endswith(suffix) for suffix in SUFFIXES)):
                if '.' in p.name:
                    continue
            elif p.is_dir():
                continue
            name = p.name
            for suffix in SUFFIXES:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            if name == 'badsyntax_pep3120':
                continue
            package_pathname = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', ImportWarning)
                    spec = finder.find_spec(name)
                    if spec is None:
                        continue
                    if spec.submodule_search_locations is not None:
                        package_pathname = spec.submodule_search_locations[0]
            except (ImportError, OSError, SyntaxError, UnicodeEncodeError):
                continue
            if package_pathname is not None:
                path_real = Path(package_pathname).resolve()
                try:
                    stat = path_real.stat()
                except OSError:
                    continue
                loaded_inode = _LoadedInode(stat.st_dev, stat.st_ino)
                if loaded_inode not in self.paths:
                    self.paths.add(loaded_inode)
                    for subname in self.find_modules(path_real):
                        if subname is None:
                            yield None
                        elif subname != '__init__':
                            yield f'{name}.{subname}'
            yield name
        yield None

    def find_all_modules(self, paths: Iterable[Path]) -> Generator[None, None, None]:
        """Return a list with all modules in `path`, which should be a list of
        directory names. If path is not given, sys.path will be used."""
        for p in paths:
            for module in self.find_modules(p):
                if module is not None:
                    self.modules.add(module)
                yield

    def find_coroutine(self) -> bool:
        if self.fully_loaded:
            return False
        try:
            next(self.find_iterator)
        except StopIteration:
            self.fully_loaded = True
        return True