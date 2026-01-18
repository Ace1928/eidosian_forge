import collections
import copy
import enum
import functools
import inspect
import pickle
import warnings
from typing import Any, Callable, Dict, List, Set, Tuple, Union
import torch
import torch._jit_internal as _jit_internal
from torch._classes import classes
from torch._jit_internal import _qualified_name
from torch.jit._builtins import _register_builtin
from torch.jit._fuser import _graph_for, _script_method_graph_for
from torch.jit._monkeytype_config import (
from torch.jit._recursive import (
from torch.jit._state import (
from torch.jit.frontend import get_default_args, get_jit_class_def, get_jit_def
from torch.nn import Module
from torch.overrides import (
from torch.package import PackageExporter, PackageImporter
from torch.utils import set_module
from ._serialization import validate_map_location
def unpackage_script_module(importer: PackageImporter, script_module_id: str) -> torch.nn.Module:
    """
    Call by ``torch.package.PackageImporter``'s Pickler's ``persistent_load`` function.

    Performs work of loading and returning a ScriptModule from a ``torch.package`` archive.
    """
    if not isinstance(importer.zip_reader, torch._C.PyTorchFileReader):
        raise RuntimeError('Loading ScriptObjects from a PackageImporter created from a directory is not supported. Use a package archive file instead.')
    cu = torch._C.CompilationUnit()
    cpp_module = torch._C._import_ir_module_from_package(cu, importer.zip_reader, importer.storage_context, validate_map_location(importer.last_map_location), script_module_id)
    return wrap_cpp_module(cpp_module)