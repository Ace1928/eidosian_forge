import sys
import ctypes
import threading
import importlib.resources as _impres
from llvmlite.binding.common import _decode_string, _is_shutting_down
from llvmlite.utils import get_library_name
def _load_lib(self):
    try:
        with _suppress_cleanup_errors(_importlib_resources_path(__name__.rpartition('.')[0], get_library_name())) as lib_path:
            self._lib_handle = ctypes.CDLL(str(lib_path))
            _ = self._lib_handle.LLVMPY_GetVersionInfo()
    except (OSError, AttributeError) as e:
        raise OSError('Could not find/load shared object file') from e