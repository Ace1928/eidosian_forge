import sys
import ctypes
import threading
import importlib.resources as _impres
from llvmlite.binding.common import _decode_string, _is_shutting_down
from llvmlite.utils import get_library_name
def _importlib_resources_path_repl(package, resource):
    """Replacement implementation of `import.resources.path` to avoid
    deprecation warning following code at importlib_resources/_legacy.py
    as suggested by https://importlib-resources.readthedocs.io/en/latest/using.html#migrating-from-legacy

    Notes on differences from importlib.resources implementation:

    The `_common.normalize_path(resource)` call is skipped because it is an
    internal API and it is unnecessary for the use here. What it does is
    ensuring `resource` is a str and that it does not contain path separators.
    """
    return _impres.as_file(_impres.files(package) / resource)