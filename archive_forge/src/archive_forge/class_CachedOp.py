import ctypes
from ..base import _LIB
from ..base import c_str_array, c_handle_array
from ..base import NDArrayHandle, CachedOpHandle
from ..base import check_call
from .. import _global_var
class CachedOp(object):
    """Cached operator handle."""
    __slots__ = ['handle', 'is_np_sym', '_monitor_callback']

    def __init__(self, sym, flags=()):
        self.handle = CachedOpHandle()
        self._monitor_callback = None
        from ..symbol.numpy._symbol import _Symbol
        self.is_np_sym = bool(isinstance(sym, _Symbol))
        check_call(_LIB.MXCreateCachedOpEx(sym.handle, len(flags), c_str_array([key for key, _ in flags]), c_str_array([str(val) for _, val in flags]), ctypes.byref(self.handle)))

    def __del__(self):
        check_call(_LIB.MXFreeCachedOp(self.handle))

    def __call__(self, *args, **kwargs):
        """ctypes implementation of imperative invoke wrapper"""
        out = kwargs.pop('out', None)
        if out is not None:
            original_output = out
            if isinstance(out, NDArrayBase):
                out = (out,)
            num_output = ctypes.c_int(len(out))
            output_vars = c_handle_array(out)
            output_vars = ctypes.cast(output_vars, ctypes.POINTER(NDArrayHandle))
        else:
            original_output = None
            output_vars = ctypes.POINTER(NDArrayHandle)()
            num_output = ctypes.c_int(0)
        if kwargs:
            raise TypeError('CachedOp.__call__ got unexpected keyword argument(s): ' + ', '.join(kwargs.keys()))
        out_stypes = ctypes.POINTER(ctypes.c_int)()
        check_call(_LIB.MXInvokeCachedOpEx(self.handle, ctypes.c_int(len(args)), c_handle_array(args), ctypes.byref(num_output), ctypes.byref(output_vars), ctypes.byref(out_stypes)))
        if original_output is not None:
            return original_output
        create_ndarray_fn = _global_var._np_ndarray_cls if self.is_np_sym else _global_var._ndarray_cls
        if num_output.value == 1:
            return create_ndarray_fn(ctypes.cast(output_vars[0], NDArrayHandle), stype=out_stypes[0])
        else:
            return [create_ndarray_fn(ctypes.cast(output_vars[i], NDArrayHandle), stype=out_stypes[i]) for i in range(num_output.value)]

    def _register_op_hook(self, callback, monitor_all=False):
        """Install callback for monitor.

        Parameters
        ----------
        callback : function
            Takes a string for node_name, string for op_name and a NDArrayHandle.
        monitor_all : bool, default False
            If true, monitor both input _imperative_invoked output, otherwise monitor output only.
        """
        cb_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p, NDArrayHandle, ctypes.c_void_p)
        if callback:
            self._monitor_callback = cb_type(_monitor_callback_wrapper(callback))
        check_call(_LIB.MXCachedOpRegisterOpHook(self.handle, self._monitor_callback, ctypes.c_int(monitor_all)))