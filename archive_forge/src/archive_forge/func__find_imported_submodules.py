import builtins
import dis
import opcode
import platform
import sys
import types
import weakref
import uuid
import threading
import typing
import warnings
from .compat import pickle
from collections import OrderedDict
from typing import ClassVar, Generic, Union, Tuple, Callable
from pickle import _getattribute
def _find_imported_submodules(code, top_level_dependencies):
    """
    Find currently imported submodules used by a function.

    Submodules used by a function need to be detected and referenced for the
    function to work correctly at depickling time. Because submodules can be
    referenced as attribute of their parent package (``package.submodule``), we
    need a special introspection technique that does not rely on GLOBAL-related
    opcodes to find references of them in a code object.

    Example:
    ```
    import concurrent.futures
    import cloudpickle
    def func():
        x = concurrent.futures.ThreadPoolExecutor
    if __name__ == '__main__':
        cloudpickle.dumps(func)
    ```
    The globals extracted by cloudpickle in the function's state include the
    concurrent package, but not its submodule (here, concurrent.futures), which
    is the module used by func. Find_imported_submodules will detect the usage
    of concurrent.futures. Saving this module alongside with func will ensure
    that calling func once depickled does not fail due to concurrent.futures
    not being imported
    """
    subimports = []
    for x in top_level_dependencies:
        if isinstance(x, types.ModuleType) and hasattr(x, '__package__') and x.__package__:
            prefix = x.__name__ + '.'
            for name in list(sys.modules):
                if name is not None and name.startswith(prefix):
                    tokens = set(name[len(prefix):].split('.'))
                    if not tokens - set(code.co_names):
                        subimports.append(sys.modules[name])
    return subimports