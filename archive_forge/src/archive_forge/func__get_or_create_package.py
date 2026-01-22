import builtins
import importlib
import importlib.machinery
import inspect
import io
import linecache
import os.path
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Callable, cast, Dict, Iterable, List, Optional, Union
from weakref import WeakValueDictionary
import torch
from torch.serialization import _get_restore_location, _maybe_decode_ascii
from ._directory_reader import DirectoryReader
from ._importlib import (
from ._mangling import demangle, PackageMangler
from ._package_unpickler import PackageUnpickler
from .file_structure_representation import _create_directory_from_file_list, Directory
from .glob_group import GlobPattern
from .importer import Importer
def _get_or_create_package(self, atoms: List[str]) -> 'Union[_PackageNode, _ExternNode]':
    cur = self.root
    for i, atom in enumerate(atoms):
        node = cur.children.get(atom, None)
        if node is None:
            node = cur.children[atom] = _PackageNode(None)
        if isinstance(node, _ExternNode):
            return node
        if isinstance(node, _ModuleNode):
            name = '.'.join(atoms[:i])
            raise ImportError(f'inconsistent module structure. module {name} is not a package, but has submodules')
        assert isinstance(node, _PackageNode)
        cur = node
    return cur