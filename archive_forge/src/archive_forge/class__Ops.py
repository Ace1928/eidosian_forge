import contextlib
import ctypes
import importlib
import inspect
import sys
import types
from typing import Any, Callable, Dict, List, Type, Union
import torch._C
import torch.utils._pytree as pytree
from torch import _utils_internal
from torch._functorch.pyfunctorch import dispatch_functorch
class _Ops(types.ModuleType):
    __file__ = '_ops.py'

    def __init__(self):
        super().__init__('torch.ops')
        self.loaded_libraries = set()
        self._higher_order_op_namespace = _PyOpNamespace('torch.ops.higher_order', _higher_order_ops)
        self._dir = []

    def __getattr__(self, name):
        if name == 'higher_order':
            return self._higher_order_op_namespace
        namespace = _OpNamespace(name)
        setattr(self, name, namespace)
        self._dir.append(name)
        return namespace

    def __iter__(self):
        return iter(self._dir)

    def import_module(self, module):
        """
        Imports a Python module that has torch.library registrations.

        Generally, to extend PyTorch with custom operators, a user will
        create a Python module whose import triggers registration of
        the custom operators via a torch.ops.load_library call or a call
        to one or more torch.library.* APIs.

        It is unexpected for Python modules to have side effects, so some
        linters and formatters will complain. Use this API to import Python
        modules that contain these torch.library side effects.

        Args:
            module (str): The name of the Python module to import

        """
        importlib.import_module(module)

    def load_library(self, path):
        """
        Loads a shared library from the given path into the current process.

        The library being loaded may run global initialization code to register
        custom operators with the PyTorch JIT runtime. This allows dynamically
        loading custom operators. For this, you should compile your operator
        and the static registration code into a shared library object, and then
        call ``torch.ops.load_library('path/to/libcustom.so')`` to load the
        shared object.

        After the library is loaded, it is added to the
        ``torch.ops.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.

        Args:
            path (str): A path to a shared library to load.
        """
        if torch._running_with_deploy():
            return
        path = _utils_internal.resolve_library_path(path)
        with dl_open_guard():
            ctypes.CDLL(path)
        self.loaded_libraries.add(path)