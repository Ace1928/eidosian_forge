import atexit
import contextlib
from enum import Enum
from errno import EBADF
from errno import ELOOP
from errno import ENOENT
from errno import ENOTDIR
import fnmatch
from functools import partial
import importlib.util
import itertools
import os
from os.path import expanduser
from os.path import expandvars
from os.path import isabs
from os.path import sep
from pathlib import Path
from pathlib import PurePath
from posixpath import sep as posix_sep
import shutil
import sys
import types
from types import ModuleType
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import uuid
import warnings
from _pytest.compat import assert_never
from _pytest.outcomes import skip
from _pytest.warning_types import PytestWarning
def _import_module_using_spec(module_name: str, module_path: Path, module_location: Path, *, insert_modules: bool) -> Optional[ModuleType]:
    """
    Tries to import a module by its canonical name, path to the .py file, and its
    parent location.

    :param insert_modules:
        If True, will call insert_missing_modules to create empty intermediate modules
        for made-up module names (when importing test files not reachable from sys.path).
        Note: we can probably drop insert_missing_modules altogether: instead of
        generating module names such as "src.tests.test_foo", which require intermediate
        empty modules, we might just as well generate unique module names like
        "src_tests_test_foo".
    """
    for meta_importer in sys.meta_path:
        spec = meta_importer.find_spec(module_name, [str(module_location)])
        if spec is not None:
            break
    else:
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is not None:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        if insert_modules:
            insert_missing_modules(sys.modules, module_name)
        return mod
    return None