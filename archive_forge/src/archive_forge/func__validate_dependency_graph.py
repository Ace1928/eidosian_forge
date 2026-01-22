import collections
import importlib.machinery
import io
import linecache
import pickletools
import platform
import types
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import (
import torch
from torch.serialization import location_tag, normalize_storage_type
from torch.types import Storage
from torch.utils.hooks import RemovableHandle
from ._digraph import DiGraph
from ._importlib import _normalize_path
from ._mangling import demangle, is_mangled
from ._package_pickler import create_pickler
from ._stdlib import is_stdlib_module
from .find_file_dependencies import find_files_source_depends_on
from .glob_group import GlobGroup, GlobPattern
from .importer import Importer, OrderedImporter, sys_importer
from _mock import MockedObject
def _validate_dependency_graph(self):
    for attrs in self.dependency_graph.nodes.values():
        if 'error' in attrs:
            raise PackagingError(self.dependency_graph, debug=self.debug)
    for pattern, pattern_info in self.patterns.items():
        if not pattern_info.allow_empty and (not pattern_info.was_matched):
            raise EmptyMatchError(f'Exporter did not match any modules to {pattern}, which was marked as allow_empty=False')