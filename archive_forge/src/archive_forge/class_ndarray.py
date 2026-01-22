import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
class ndarray(object):
    """
    Wrapper around cupy.ndarray
    Supports cupy.ndarray.__init__ as well as,
    gets initialized with a cupy ndarray.
    """
    __doc__ = np.ndarray.__doc__

    def __new__(cls, *args, **kwargs):
        """
        If `_initial_array` and `_supports_cupy` are arguments,
        initialize cls(ndarray).
        Else get cupy.ndarray from provided arguments,
        then initialize cls(ndarray).
        """
        _initial_array = kwargs.get('_initial_array', None)
        if _initial_array is not None:
            return object.__new__(cls)
        cupy_ndarray_init = cp.ndarray(*args, **kwargs)
        return cls(_initial_array=cupy_ndarray_init, _supports_cupy=True)

    def __init__(self, *args, **kwargs):
        """
        Args:
            _initial_array (None, cp.ndarray/np.ndarray(including variants)):
                If _initial_array is None, object is not initialized.
                Otherwise, _initial_array (ndarray) would be set to
                _cupy_array and/or _numpy_array depending upon _supports_cupy.
            _supports_cupy (bool): If _supports_cupy is True, _initial_array
                is set as _cupy_array and _numpy_array.
                Otherwise, _initial_array is set as only _numpy_array.

        Attributes:
            _cupy_array (None or cp.ndarray): ndarray fully compatible with
                CuPy. This will be always set to a ndarray in GPU.
            _numpy_array (None or np.ndarray(including variants)): ndarray not
                supported by CuPy. Such as np.ndarray (where dtype is not in
                '?bhilqBHILQefdFD') and it's variants. This will be always set
                to a ndarray in CPU.
            _supports_cupy (bool): If _supports_cupy is True, data of array
                will contain in _cupy_array and _numpy_array.
                Else only _numpy_array will have the data.
        """
        _supports_cupy = kwargs.pop('_supports_cupy', None)
        _initial_array = kwargs.pop('_initial_array', None)
        if _initial_array is None:
            return
        self._cupy_array = None
        self._numpy_array = None
        self.base = None
        self._supports_cupy = _supports_cupy
        assert isinstance(_initial_array, (cp.ndarray, np.ndarray))
        if _supports_cupy:
            if type(_initial_array) is cp.ndarray:
                self._cupy_array = _initial_array
                self._remember_numpy = False
            else:
                self._numpy_array = _initial_array
                self._remember_numpy = True
        else:
            self._numpy_array = _initial_array

    @classmethod
    def _store_array_from_cupy(cls, array):
        return cls(_initial_array=array, _supports_cupy=True)

    @classmethod
    def _store_array_from_numpy(cls, array):
        if type(array) is np.ndarray and array.dtype.kind in '?bhilqBHILQefdFD':
            return cls(_initial_array=array, _supports_cupy=True)
        return cls(_initial_array=array, _supports_cupy=False)

    @property
    def dtype(self):
        if self._supports_cupy and (not self._remember_numpy):
            return self._cupy_array.dtype
        return self._numpy_array.dtype

    def __getattr__(self, attr):
        """
        Catches attributes corresponding to ndarray.

        Args:
            attr (str): Attribute of ndarray class.

        Returns:
            (_RecursiveAttr object, self._array.attr):
            Returns_RecursiveAttr object with numpy_object, cupy_object.
            Returns self._array.attr if attr is not callable.
        """
        if self._supports_cupy:
            cupy_object = getattr(cp.ndarray, attr, None)
            numpy_object = getattr(np.ndarray, attr)
        else:
            cupy_object = None
            numpy_object = getattr(self._numpy_array.__class__, attr)
        if not callable(numpy_object):
            if self._supports_cupy:
                if self._remember_numpy:
                    self._update_cupy_array()
                return getattr(self._cupy_array, attr)
            return getattr(self._numpy_array, attr)
        return _RecursiveAttr(numpy_object, cupy_object, self)

    def _get_cupy_array(self):
        """
        Returns _cupy_array (cupy.ndarray) of ndarray object. And marks
        self(ndarray) and it's base (if exist) as numpy not up-to-date.
        """
        base = self.base
        if base is not None:
            base._remember_numpy = False
        self._remember_numpy = False
        return self._cupy_array

    def _get_numpy_array(self):
        """
        Returns _numpy_array (ex: np.ndarray, numpy.ma.MaskedArray,
        numpy.chararray etc.) of ndarray object. And marks self(ndarray)
        and it's base (if exist) as numpy up-to-date.
        """
        base = self.base
        if base is not None and base._supports_cupy:
            base._remember_numpy = True
        if self._supports_cupy:
            self._remember_numpy = True
        return self._numpy_array

    def _update_numpy_array(self):
        """
        Updates _numpy_array from _cupy_array.
        To be executed before calling numpy function.
        """
        base = self.base
        _type = np.ndarray if self._supports_cupy else self._numpy_array.__class__
        if self._supports_cupy:
            if base is None:
                if not self._remember_numpy:
                    if self._numpy_array is None:
                        self._numpy_array = cp.asnumpy(self._cupy_array)
                    else:
                        self._cupy_array.get(out=self._numpy_array)
            elif not base._remember_numpy:
                base._update_numpy_array()
                if self._numpy_array is None:
                    self._numpy_array = base._numpy_array.view(type=_type)
                    self._numpy_array.shape = self._cupy_array.shape
                    self._numpy_array.strides = self._cupy_array.strides
        elif base is not None:
            assert base._supports_cupy
            if not base._remember_numpy:
                base._update_numpy_array()

    def _update_cupy_array(self):
        """
        Updates _cupy_array from _numpy_array.
        To be executed before calling cupy function.
        """
        base = self.base
        if base is None:
            if self._remember_numpy:
                if self._cupy_array is None:
                    self._cupy_array = cp.array(self._numpy_array)
                else:
                    self._cupy_array[:] = self._numpy_array
        elif base._remember_numpy:
            base._update_cupy_array()