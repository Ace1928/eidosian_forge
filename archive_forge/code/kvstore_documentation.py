import pickle
import ctypes
import os
from ..ndarray import NDArray
from ..ndarray import _ndarray_cls
from ..base import _LIB, c_str
from ..base import check_call, mx_uint, py_str
from ..base import NDArrayHandle, KVStoreHandle
from .. import optimizer as opt
from .base import _ctype_key_value, _ctype_dict, KVStoreBase
Sends a command to all server nodes.

        Sending command to a server node will cause that server node to invoke
        ``KVStoreServer.controller`` to execute the command.

        This function returns after the command has been executed on all server
        nodes.

        Parameters
        ----------
        head : int
            the head of the command.
        body : str
            the body of the command.
        