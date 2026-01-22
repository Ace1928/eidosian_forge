import traceback
import warnings
import collections
from array import array
from threading import Lock
import ctypes
from ctypes import CFUNCTYPE, POINTER, Structure, pointer
from ctypes import c_void_p, c_int, c_char, c_char_p, cast, c_bool
from .base import _LIB, check_call, MXCallbackList, c_array, c_array_buf, mx_int, OpHandle
from .base import c_str, mx_uint, mx_float, ctypes2numpy_shared, NDArrayHandle, py_str
from . import symbol, context
from .ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID, _STORAGE_TYPE_ID_TO_STR
from .ndarray.ndarray import _STORAGE_TYPE_UNDEFINED, _STORAGE_TYPE_DEFAULT
from .ndarray.ndarray import _STORAGE_TYPE_CSR, _STORAGE_TYPE_ROW_SPARSE
from .ndarray import _ndarray_cls
from .numpy.multiarray import _np_ndarray_cls
from .util import is_np_array
class PythonOp(object):
    """Base class for operators implemented in Python.

    Parameters
    ----------
    need_top_grad : bool
        the default need_top_grad() function returns this value.
    """
    _ref_holder = []

    def __init__(self, need_top_grad=True):
        self.info_ = None
        self.need_top_grad_ = need_top_grad
        warnings.warn('PythonOp has been deprecated. Please use CustomOp')

    def __call__(self, *args, **kwargs):
        return self.get_symbol(*args, **kwargs)

    def get_symbol(self, *args, **kwargs):
        """Create a symbol from numpy operator.
        This should only be called once per instance if the operator contains
        internal states.

        Parameters
        ----------
        args : list
            a list of input arguments (symbols).

        Returns
        -------
        sym : mxnet.symbol.Symbol
        """
        raise NotImplementedError('Must override this')

    def forward(self, in_data, out_data):
        """Forward interface. Override to create new operators.

        Parameters
        ----------
        in_data, out_data: list
            input and output for forward. See document for
            corresponding arguments of Operator::Forward
        """
        out_data[0][:] = in_data[0]

    def backward(self, out_grad, in_data, out_data, in_grad):
        """Backward interface. Can override when creating new operators.

        Parameters
        ----------
        out_grad, in_data, out_data, in_grad : list
            input and output for backward. See document for
            corresponding arguments of Operator::Backward
        """
        in_grad[0][:] = 1.0

    def infer_shape(self, in_shape):
        """Interface for ``infer_shape``. Can override when creating new operators.

        Parameters
        ----------
        in_shape : list
            List of argument shapes in the same order as
            declared in list_arguments.

        Returns
        -------
        in_shape : list
            List of argument shapes. Can be modified from in_shape.
        out_shape : list
            List of output shapes calculated from in_shape,
            in the same order as declared in list_arguments.
        """
        return (in_shape, [in_shape[0]])

    def list_outputs(self):
        """Interface for ``list_outputs``. Can override when creating new operators.

        Returns
        -------
        outputs : list
            List of output blob names.
        """
        return ['output']

    def list_arguments(self):
        """Interface for ``list_arguments``. Can override when creating new operators.

        Returns
        -------
        in_shape : list
            list of argument shapes in the same order as
            declared in list_arguments.
        """
        return ['data']

    def need_top_grad(self):
        """Whether this operator needs out_grad for backward.

        Returns
        -------
        need_top_grad : bool
            Whether this operator needs out_grad for backward.
            Should be set to False for loss layers.
        """
        return self.need_top_grad_