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
def _get_dependencies(self, src: str, module_name: str, is_package: bool) -> List[str]:
    """Return all modules that this source code depends on.

        Dependencies are found by scanning the source code for import-like statements.

        Arguments:
            src: The Python source code to analyze for dependencies.
            module_name: The name of the module that ``src`` corresponds to.
            is_package: Whether this module should be treated as a package.
                See :py:meth:`save_source_string` for more info.

        Returns:
            A list containing modules detected as direct dependencies in
            ``src``.  The items in the list are guaranteed to be unique.
        """
    package_name = module_name if is_package else module_name.rsplit('.', maxsplit=1)[0]
    try:
        dep_pairs = find_files_source_depends_on(src, package_name)
    except Exception as e:
        self.dependency_graph.add_node(module_name, error=PackagingErrorReason.DEPENDENCY_RESOLUTION_FAILED, error_context=str(e))
        return []
    dependencies = {}
    for dep_module_name, dep_module_obj in dep_pairs:
        if dep_module_obj is not None:
            possible_submodule = f'{dep_module_name}.{dep_module_obj}'
            if self._module_exists(possible_submodule):
                dependencies[possible_submodule] = True
                continue
        if self._module_exists(dep_module_name):
            dependencies[dep_module_name] = True
    return list(dependencies.keys())