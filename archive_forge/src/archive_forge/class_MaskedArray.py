import builtins
import inspect
import operator
import warnings
import textwrap
import re
from functools import reduce
import numpy as np
import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy.core import multiarray as mu
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue
from numpy import array as narray
from numpy.lib.function_base import angle
from numpy.compat import (
from numpy import expand_dims
from numpy.core.numeric import normalize_axis_tuple
frombuffer = _convert2ma(
fromfunction = _convert2ma(
class MaskedArray(ndarray):
    """
    An array class with possibly masked values.

    Masked values of True exclude the corresponding element from any
    computation.

    Construction::

      x = MaskedArray(data, mask=nomask, dtype=None, copy=False, subok=True,
                      ndmin=0, fill_value=None, keep_mask=True, hard_mask=None,
                      shrink=True, order=None)

    Parameters
    ----------
    data : array_like
        Input data.
    mask : sequence, optional
        Mask. Must be convertible to an array of booleans with the same
        shape as `data`. True indicates a masked (i.e. invalid) data.
    dtype : dtype, optional
        Data type of the output.
        If `dtype` is None, the type of the data argument (``data.dtype``)
        is used. If `dtype` is not None and different from ``data.dtype``,
        a copy is performed.
    copy : bool, optional
        Whether to copy the input data (True), or to use a reference instead.
        Default is False.
    subok : bool, optional
        Whether to return a subclass of `MaskedArray` if possible (True) or a
        plain `MaskedArray`. Default is True.
    ndmin : int, optional
        Minimum number of dimensions. Default is 0.
    fill_value : scalar, optional
        Value used to fill in the masked values when necessary.
        If None, a default based on the data-type is used.
    keep_mask : bool, optional
        Whether to combine `mask` with the mask of the input data, if any
        (True), or to use only `mask` for the output (False). Default is True.
    hard_mask : bool, optional
        Whether to use a hard mask or not. With a hard mask, masked values
        cannot be unmasked. Default is False.
    shrink : bool, optional
        Whether to force compression of an empty mask. Default is True.
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C', then the array
        will be in C-contiguous order (last-index varies the fastest).
        If order is 'F', then the returned array will be in
        Fortran-contiguous order (first-index varies the fastest).
        If order is 'A' (default), then the returned array may be
        in any order (either C-, Fortran-contiguous, or even discontiguous),
        unless a copy is required, in which case it will be C-contiguous.

    Examples
    --------

    The ``mask`` can be initialized with an array of boolean values
    with the same shape as ``data``.

    >>> data = np.arange(6).reshape((2, 3))
    >>> np.ma.MaskedArray(data, mask=[[False, True, False],
    ...                               [False, False, True]])
    masked_array(
      data=[[0, --, 2],
            [3, 4, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)

    Alternatively, the ``mask`` can be initialized to homogeneous boolean
    array with the same shape as ``data`` by passing in a scalar
    boolean value:

    >>> np.ma.MaskedArray(data, mask=False)
    masked_array(
      data=[[0, 1, 2],
            [3, 4, 5]],
      mask=[[False, False, False],
            [False, False, False]],
      fill_value=999999)

    >>> np.ma.MaskedArray(data, mask=True)
    masked_array(
      data=[[--, --, --],
            [--, --, --]],
      mask=[[ True,  True,  True],
            [ True,  True,  True]],
      fill_value=999999,
      dtype=int64)

    .. note::
        The recommended practice for initializing ``mask`` with a scalar
        boolean value is to use ``True``/``False`` rather than
        ``np.True_``/``np.False_``. The reason is :attr:`nomask`
        is represented internally as ``np.False_``.

        >>> np.False_ is np.ma.nomask
        True

    """
    __array_priority__ = 15
    _defaultmask = nomask
    _defaulthardmask = False
    _baseclass = ndarray
    _print_width = 100
    _print_width_1d = 1500

    def __new__(cls, data=None, mask=nomask, dtype=None, copy=False, subok=True, ndmin=0, fill_value=None, keep_mask=True, hard_mask=None, shrink=True, order=None):
        """
        Create a new masked array from scratch.

        Notes
        -----
        A masked array can also be created by taking a .view(MaskedArray).

        """
        _data = np.array(data, dtype=dtype, copy=copy, order=order, subok=True, ndmin=ndmin)
        _baseclass = getattr(data, '_baseclass', type(_data))
        if isinstance(data, MaskedArray) and data.shape != _data.shape:
            copy = True
        if isinstance(data, cls) and subok and (not isinstance(data, MaskedConstant)):
            _data = ndarray.view(_data, type(data))
        else:
            _data = ndarray.view(_data, cls)
        if hasattr(data, '_mask') and (not isinstance(data, ndarray)):
            _data._mask = data._mask
        mdtype = make_mask_descr(_data.dtype)
        if mask is nomask:
            if not keep_mask:
                if shrink:
                    _data._mask = nomask
                else:
                    _data._mask = np.zeros(_data.shape, dtype=mdtype)
            elif isinstance(data, (tuple, list)):
                try:
                    mask = np.array([getmaskarray(np.asanyarray(m, dtype=_data.dtype)) for m in data], dtype=mdtype)
                except (ValueError, TypeError):
                    mask = nomask
                if mdtype == MaskType and mask.any():
                    _data._mask = mask
                    _data._sharedmask = False
            else:
                _data._sharedmask = not copy
                if copy:
                    _data._mask = _data._mask.copy()
                    if getmask(data) is not nomask:
                        if data._mask.shape != data.shape:
                            data._mask.shape = data.shape
        else:
            if mask is None:
                mask = False
            if mask is True and mdtype == MaskType:
                mask = np.ones(_data.shape, dtype=mdtype)
            elif mask is False and mdtype == MaskType:
                mask = np.zeros(_data.shape, dtype=mdtype)
            else:
                try:
                    mask = np.array(mask, copy=copy, dtype=mdtype)
                except TypeError:
                    mask = np.array([tuple([m] * len(mdtype)) for m in mask], dtype=mdtype)
            if mask.shape != _data.shape:
                nd, nm = (_data.size, mask.size)
                if nm == 1:
                    mask = np.resize(mask, _data.shape)
                elif nm == nd:
                    mask = np.reshape(mask, _data.shape)
                else:
                    msg = 'Mask and data not compatible: data size is %i, ' + 'mask size is %i.'
                    raise MaskError(msg % (nd, nm))
                copy = True
            if _data._mask is nomask:
                _data._mask = mask
                _data._sharedmask = not copy
            elif not keep_mask:
                _data._mask = mask
                _data._sharedmask = not copy
            else:
                if _data.dtype.names is not None:

                    def _recursive_or(a, b):
                        """do a|=b on each field of a, recursively"""
                        for name in a.dtype.names:
                            af, bf = (a[name], b[name])
                            if af.dtype.names is not None:
                                _recursive_or(af, bf)
                            else:
                                af |= bf
                    _recursive_or(_data._mask, mask)
                else:
                    _data._mask = np.logical_or(mask, _data._mask)
                _data._sharedmask = False
        if fill_value is None:
            fill_value = getattr(data, '_fill_value', None)
        if fill_value is not None:
            _data._fill_value = _check_fill_value(fill_value, _data.dtype)
        if hard_mask is None:
            _data._hardmask = getattr(data, '_hardmask', False)
        else:
            _data._hardmask = hard_mask
        _data._baseclass = _baseclass
        return _data

    def _update_from(self, obj):
        """
        Copies some attributes of obj to self.

        """
        if isinstance(obj, ndarray):
            _baseclass = type(obj)
        else:
            _baseclass = ndarray
        _optinfo = {}
        _optinfo.update(getattr(obj, '_optinfo', {}))
        _optinfo.update(getattr(obj, '_basedict', {}))
        if not isinstance(obj, MaskedArray):
            _optinfo.update(getattr(obj, '__dict__', {}))
        _dict = dict(_fill_value=getattr(obj, '_fill_value', None), _hardmask=getattr(obj, '_hardmask', False), _sharedmask=getattr(obj, '_sharedmask', False), _isfield=getattr(obj, '_isfield', False), _baseclass=getattr(obj, '_baseclass', _baseclass), _optinfo=_optinfo, _basedict=_optinfo)
        self.__dict__.update(_dict)
        self.__dict__.update(_optinfo)
        return

    def __array_finalize__(self, obj):
        """
        Finalizes the masked array.

        """
        self._update_from(obj)
        if isinstance(obj, ndarray):
            if obj.dtype.names is not None:
                _mask = getmaskarray(obj)
            else:
                _mask = getmask(obj)
            if _mask is not nomask and obj.__array_interface__['data'][0] != self.__array_interface__['data'][0]:
                if self.dtype == obj.dtype:
                    _mask_dtype = _mask.dtype
                else:
                    _mask_dtype = make_mask_descr(self.dtype)
                if self.flags.c_contiguous:
                    order = 'C'
                elif self.flags.f_contiguous:
                    order = 'F'
                else:
                    order = 'K'
                _mask = _mask.astype(_mask_dtype, order)
            else:
                _mask = _mask.view()
        else:
            _mask = nomask
        self._mask = _mask
        if self._mask is not nomask:
            try:
                self._mask.shape = self.shape
            except ValueError:
                self._mask = nomask
            except (TypeError, AttributeError):
                pass
        if self._fill_value is not None:
            self._fill_value = _check_fill_value(self._fill_value, self.dtype)
        elif self.dtype.names is not None:
            self._fill_value = _check_fill_value(None, self.dtype)

    def __array_wrap__(self, obj, context=None):
        """
        Special hook for ufuncs.

        Wraps the numpy array and sets the mask according to context.

        """
        if obj is self:
            result = obj
        else:
            result = obj.view(type(self))
            result._update_from(self)
        if context is not None:
            result._mask = result._mask.copy()
            func, args, out_i = context
            input_args = args[:func.nin]
            m = reduce(mask_or, [getmaskarray(arg) for arg in input_args])
            domain = ufunc_domain.get(func, None)
            if domain is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    d = filled(domain(*input_args), True)
                if d.any():
                    try:
                        fill_value = ufunc_fills[func][-1]
                    except TypeError:
                        fill_value = ufunc_fills[func]
                    except KeyError:
                        fill_value = self.fill_value
                    np.copyto(result, fill_value, where=d)
                    if m is nomask:
                        m = d
                    else:
                        m = m | d
            if result is not self and result.shape == () and m:
                return masked
            else:
                result._mask = m
                result._sharedmask = False
        return result

    def view(self, dtype=None, type=None, fill_value=None):
        """
        Return a view of the MaskedArray data.

        Parameters
        ----------
        dtype : data-type or ndarray sub-class, optional
            Data-type descriptor of the returned view, e.g., float32 or int16.
            The default, None, results in the view having the same data-type
            as `a`. As with ``ndarray.view``, dtype can also be specified as
            an ndarray sub-class, which then specifies the type of the
            returned object (this is equivalent to setting the ``type``
            parameter).
        type : Python type, optional
            Type of the returned view, either ndarray or a subclass.  The
            default None results in type preservation.
        fill_value : scalar, optional
            The value to use for invalid entries (None by default).
            If None, then this argument is inferred from the passed `dtype`, or
            in its absence the original array, as discussed in the notes below.

        See Also
        --------
        numpy.ndarray.view : Equivalent method on ndarray object.

        Notes
        -----

        ``a.view()`` is used two different ways:

        ``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view
        of the array's memory with a different data-type.  This can cause a
        reinterpretation of the bytes of memory.

        ``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just
        returns an instance of `ndarray_subclass` that looks at the same array
        (same shape, dtype, etc.)  This does not cause a reinterpretation of the
        memory.

        If `fill_value` is not specified, but `dtype` is specified (and is not
        an ndarray sub-class), the `fill_value` of the MaskedArray will be
        reset. If neither `fill_value` nor `dtype` are specified (or if
        `dtype` is an ndarray sub-class), then the fill value is preserved.
        Finally, if `fill_value` is specified, but `dtype` is not, the fill
        value is set to the specified value.

        For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
        bytes per entry than the previous dtype (for example, converting a
        regular array to a structured array), then the behavior of the view
        cannot be predicted just from the superficial appearance of ``a`` (shown
        by ``print(a)``). It also depends on exactly how ``a`` is stored in
        memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus
        defined as a slice or transpose, etc., the view may give different
        results.
        """
        if dtype is None:
            if type is None:
                output = ndarray.view(self)
            else:
                output = ndarray.view(self, type)
        elif type is None:
            try:
                if issubclass(dtype, ndarray):
                    output = ndarray.view(self, dtype)
                    dtype = None
                else:
                    output = ndarray.view(self, dtype)
            except TypeError:
                output = ndarray.view(self, dtype)
        else:
            output = ndarray.view(self, dtype, type)
        if getmask(output) is not nomask:
            output._mask = output._mask.view()
        if getattr(output, '_fill_value', None) is not None:
            if fill_value is None:
                if dtype is None:
                    pass
                else:
                    output._fill_value = None
            else:
                output.fill_value = fill_value
        return output

    def __getitem__(self, indx):
        """
        x.__getitem__(y) <==> x[y]

        Return the item described by i, as a masked array.

        """
        dout = self.data[indx]
        _mask = self._mask

        def _is_scalar(m):
            return not isinstance(m, np.ndarray)

        def _scalar_heuristic(arr, elem):
            """
            Return whether `elem` is a scalar result of indexing `arr`, or None
            if undecidable without promoting nomask to a full mask
            """
            if not isinstance(elem, np.ndarray):
                return True
            elif arr.dtype.type is np.object_:
                if arr.dtype is not elem.dtype:
                    return True
            elif type(arr).__getitem__ == ndarray.__getitem__:
                return False
            return None
        if _mask is not nomask:
            mout = _mask[indx]
            scalar_expected = _is_scalar(mout)
        else:
            mout = nomask
            scalar_expected = _scalar_heuristic(self.data, dout)
            if scalar_expected is None:
                scalar_expected = _is_scalar(getmaskarray(self)[indx])
        if scalar_expected:
            if isinstance(dout, np.void):
                return mvoid(dout, mask=mout, hardmask=self._hardmask)
            elif self.dtype.type is np.object_ and isinstance(dout, np.ndarray) and (dout is not masked):
                if mout:
                    return MaskedArray(dout, mask=True)
                else:
                    return dout
            elif mout:
                return masked
            else:
                return dout
        else:
            dout = dout.view(type(self))
            dout._update_from(self)
            if is_string_or_list_of_strings(indx):
                if self._fill_value is not None:
                    dout._fill_value = self._fill_value[indx]
                    if not isinstance(dout._fill_value, np.ndarray):
                        raise RuntimeError('Internal NumPy error.')
                    if dout._fill_value.ndim > 0:
                        if not (dout._fill_value == dout._fill_value.flat[0]).all():
                            warnings.warn(f'Upon accessing multidimensional field {indx!s}, need to keep dimensionality of fill_value at 0. Discarding heterogeneous fill_value and setting all to {dout._fill_value[0]!s}.', stacklevel=2)
                        dout._fill_value = dout._fill_value.flat[0:1].squeeze(axis=0)
                dout._isfield = True
            if mout is not nomask:
                dout._mask = reshape(mout, dout.shape)
                dout._sharedmask = True
        return dout

    @np.errstate(over='ignore', invalid='ignore')
    def __setitem__(self, indx, value):
        """
        x.__setitem__(i, y) <==> x[i]=y

        Set item described by index. If value is masked, masks those
        locations.

        """
        if self is masked:
            raise MaskError('Cannot alter the masked element.')
        _data = self._data
        _mask = self._mask
        if isinstance(indx, str):
            _data[indx] = value
            if _mask is nomask:
                self._mask = _mask = make_mask_none(self.shape, self.dtype)
            _mask[indx] = getmask(value)
            return
        _dtype = _data.dtype
        if value is masked:
            if _mask is nomask:
                _mask = self._mask = make_mask_none(self.shape, _dtype)
            if _dtype.names is not None:
                _mask[indx] = tuple([True] * len(_dtype.names))
            else:
                _mask[indx] = True
            return
        dval = getattr(value, '_data', value)
        mval = getmask(value)
        if _dtype.names is not None and mval is nomask:
            mval = tuple([False] * len(_dtype.names))
        if _mask is nomask:
            _data[indx] = dval
            if mval is not nomask:
                _mask = self._mask = make_mask_none(self.shape, _dtype)
                _mask[indx] = mval
        elif not self._hardmask:
            if isinstance(indx, masked_array) and (not isinstance(value, masked_array)):
                _data[indx.data] = dval
            else:
                _data[indx] = dval
                _mask[indx] = mval
        elif hasattr(indx, 'dtype') and indx.dtype == MaskType:
            indx = indx * umath.logical_not(_mask)
            _data[indx] = dval
        else:
            if _dtype.names is not None:
                err_msg = "Flexible 'hard' masks are not yet supported."
                raise NotImplementedError(err_msg)
            mindx = mask_or(_mask[indx], mval, copy=True)
            dindx = self._data[indx]
            if dindx.size > 1:
                np.copyto(dindx, dval, where=~mindx)
            elif mindx is nomask:
                dindx = dval
            _data[indx] = dindx
            _mask[indx] = mindx
        return

    @property
    def dtype(self):
        return super().dtype

    @dtype.setter
    def dtype(self, dtype):
        super(MaskedArray, type(self)).dtype.__set__(self, dtype)
        if self._mask is not nomask:
            self._mask = self._mask.view(make_mask_descr(dtype), ndarray)
            try:
                self._mask.shape = self.shape
            except (AttributeError, TypeError):
                pass

    @property
    def shape(self):
        return super().shape

    @shape.setter
    def shape(self, shape):
        super(MaskedArray, type(self)).shape.__set__(self, shape)
        if getmask(self) is not nomask:
            self._mask.shape = self.shape

    def __setmask__(self, mask, copy=False):
        """
        Set the mask.

        """
        idtype = self.dtype
        current_mask = self._mask
        if mask is masked:
            mask = True
        if current_mask is nomask:
            if mask is nomask:
                return
            current_mask = self._mask = make_mask_none(self.shape, idtype)
        if idtype.names is None:
            if self._hardmask:
                current_mask |= mask
            elif isinstance(mask, (int, float, np.bool_, np.number)):
                current_mask[...] = mask
            else:
                current_mask.flat = mask
        else:
            mdtype = current_mask.dtype
            mask = np.array(mask, copy=False)
            if not mask.ndim:
                if mask.dtype.kind == 'b':
                    mask = np.array(tuple([mask.item()] * len(mdtype)), dtype=mdtype)
                else:
                    mask = mask.astype(mdtype)
            else:
                try:
                    mask = np.array(mask, copy=copy, dtype=mdtype)
                except TypeError:
                    mask = np.array([tuple([m] * len(mdtype)) for m in mask], dtype=mdtype)
            if self._hardmask:
                for n in idtype.names:
                    current_mask[n] |= mask[n]
            elif isinstance(mask, (int, float, np.bool_, np.number)):
                current_mask[...] = mask
            else:
                current_mask.flat = mask
        if current_mask.shape:
            current_mask.shape = self.shape
        return
    _set_mask = __setmask__

    @property
    def mask(self):
        """ Current mask. """
        return self._mask.view()

    @mask.setter
    def mask(self, value):
        self.__setmask__(value)

    @property
    def recordmask(self):
        """
        Get or set the mask of the array if it has no named fields. For
        structured arrays, returns a ndarray of booleans where entries are
        ``True`` if **all** the fields are masked, ``False`` otherwise:

        >>> x = np.ma.array([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
        ...         mask=[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
        ...        dtype=[('a', int), ('b', int)])
        >>> x.recordmask
        array([False, False,  True, False, False])
        """
        _mask = self._mask.view(ndarray)
        if _mask.dtype.names is None:
            return _mask
        return np.all(flatten_structured_array(_mask), axis=-1)

    @recordmask.setter
    def recordmask(self, mask):
        raise NotImplementedError('Coming soon: setting the mask per records!')

    def harden_mask(self):
        """
        Force the mask to hard, preventing unmasking by assignment.

        Whether the mask of a masked array is hard or soft is determined by
        its `~ma.MaskedArray.hardmask` property. `harden_mask` sets
        `~ma.MaskedArray.hardmask` to ``True`` (and returns the modified
        self).

        See Also
        --------
        ma.MaskedArray.hardmask
        ma.MaskedArray.soften_mask

        """
        self._hardmask = True
        return self

    def soften_mask(self):
        """
        Force the mask to soft (default), allowing unmasking by assignment.

        Whether the mask of a masked array is hard or soft is determined by
        its `~ma.MaskedArray.hardmask` property. `soften_mask` sets
        `~ma.MaskedArray.hardmask` to ``False`` (and returns the modified
        self).

        See Also
        --------
        ma.MaskedArray.hardmask
        ma.MaskedArray.harden_mask

        """
        self._hardmask = False
        return self

    @property
    def hardmask(self):
        """
        Specifies whether values can be unmasked through assignments.

        By default, assigning definite values to masked array entries will
        unmask them.  When `hardmask` is ``True``, the mask will not change
        through assignments.

        See Also
        --------
        ma.MaskedArray.harden_mask
        ma.MaskedArray.soften_mask

        Examples
        --------
        >>> x = np.arange(10)
        >>> m = np.ma.masked_array(x, x>5)
        >>> assert not m.hardmask

        Since `m` has a soft mask, assigning an element value unmasks that
        element:

        >>> m[8] = 42
        >>> m
        masked_array(data=[0, 1, 2, 3, 4, 5, --, --, 42, --],
                     mask=[False, False, False, False, False, False,
                           True, True, False, True],
               fill_value=999999)

        After hardening, the mask is not affected by assignments:

        >>> hardened = np.ma.harden_mask(m)
        >>> assert m.hardmask and hardened is m
        >>> m[:] = 23
        >>> m
        masked_array(data=[23, 23, 23, 23, 23, 23, --, --, 23, --],
                     mask=[False, False, False, False, False, False,
                           True, True, False, True],
               fill_value=999999)

        """
        return self._hardmask

    def unshare_mask(self):
        """
        Copy the mask and set the `sharedmask` flag to ``False``.

        Whether the mask is shared between masked arrays can be seen from
        the `sharedmask` property. `unshare_mask` ensures the mask is not
        shared. A copy of the mask is only made if it was shared.

        See Also
        --------
        sharedmask

        """
        if self._sharedmask:
            self._mask = self._mask.copy()
            self._sharedmask = False
        return self

    @property
    def sharedmask(self):
        """ Share status of the mask (read-only). """
        return self._sharedmask

    def shrink_mask(self):
        """
        Reduce a mask to nomask when possible.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        >>> x = np.ma.array([[1,2 ], [3, 4]], mask=[0]*4)
        >>> x.mask
        array([[False, False],
               [False, False]])
        >>> x.shrink_mask()
        masked_array(
          data=[[1, 2],
                [3, 4]],
          mask=False,
          fill_value=999999)
        >>> x.mask
        False

        """
        self._mask = _shrink_mask(self._mask)
        return self

    @property
    def baseclass(self):
        """ Class of the underlying data (read-only). """
        return self._baseclass

    def _get_data(self):
        """
        Returns the underlying data, as a view of the masked array.

        If the underlying data is a subclass of :class:`numpy.ndarray`, it is
        returned as such.

        >>> x = np.ma.array(np.matrix([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
        >>> x.data
        matrix([[1, 2],
                [3, 4]])

        The type of the data can be accessed through the :attr:`baseclass`
        attribute.
        """
        return ndarray.view(self, self._baseclass)
    _data = property(fget=_get_data)
    data = property(fget=_get_data)

    @property
    def flat(self):
        """ Return a flat iterator, or set a flattened version of self to value. """
        return MaskedIterator(self)

    @flat.setter
    def flat(self, value):
        y = self.ravel()
        y[:] = value

    @property
    def fill_value(self):
        """
        The filling value of the masked array is a scalar. When setting, None
        will set to a default based on the data type.

        Examples
        --------
        >>> for dt in [np.int32, np.int64, np.float64, np.complex128]:
        ...     np.ma.array([0, 1], dtype=dt).get_fill_value()
        ...
        999999
        999999
        1e+20
        (1e+20+0j)

        >>> x = np.ma.array([0, 1.], fill_value=-np.inf)
        >>> x.fill_value
        -inf
        >>> x.fill_value = np.pi
        >>> x.fill_value
        3.1415926535897931 # may vary

        Reset to default:

        >>> x.fill_value = None
        >>> x.fill_value
        1e+20

        """
        if self._fill_value is None:
            self._fill_value = _check_fill_value(None, self.dtype)
        if isinstance(self._fill_value, ndarray):
            return self._fill_value[()]
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value=None):
        target = _check_fill_value(value, self.dtype)
        if not target.ndim == 0:
            warnings.warn('Non-scalar arrays for the fill value are deprecated. Use arrays with scalar values instead. The filled function still supports any array as `fill_value`.', DeprecationWarning, stacklevel=2)
        _fill_value = self._fill_value
        if _fill_value is None:
            self._fill_value = target
        else:
            _fill_value[()] = target
    get_fill_value = fill_value.fget
    set_fill_value = fill_value.fset

    def filled(self, fill_value=None):
        """
        Return a copy of self, with masked values filled with a given value.
        **However**, if there are no masked values to fill, self will be
        returned instead as an ndarray.

        Parameters
        ----------
        fill_value : array_like, optional
            The value to use for invalid entries. Can be scalar or non-scalar.
            If non-scalar, the resulting ndarray must be broadcastable over
            input array. Default is None, in which case, the `fill_value`
            attribute of the array is used instead.

        Returns
        -------
        filled_array : ndarray
            A copy of ``self`` with invalid entries replaced by *fill_value*
            (be it the function argument or the attribute of ``self``), or
            ``self`` itself as an ndarray if there are no invalid entries to
            be replaced.

        Notes
        -----
        The result is **not** a MaskedArray!

        Examples
        --------
        >>> x = np.ma.array([1,2,3,4,5], mask=[0,0,1,0,1], fill_value=-999)
        >>> x.filled()
        array([   1,    2, -999,    4, -999])
        >>> x.filled(fill_value=1000)
        array([   1,    2, 1000,    4, 1000])
        >>> type(x.filled())
        <class 'numpy.ndarray'>

        Subclassing is preserved. This means that if, e.g., the data part of
        the masked array is a recarray, `filled` returns a recarray:

        >>> x = np.array([(-1, 2), (-3, 4)], dtype='i8,i8').view(np.recarray)
        >>> m = np.ma.array(x, mask=[(True, False), (False, True)])
        >>> m.filled()
        rec.array([(999999,      2), (    -3, 999999)],
                  dtype=[('f0', '<i8'), ('f1', '<i8')])
        """
        m = self._mask
        if m is nomask:
            return self._data
        if fill_value is None:
            fill_value = self.fill_value
        else:
            fill_value = _check_fill_value(fill_value, self.dtype)
        if self is masked_singleton:
            return np.asanyarray(fill_value)
        if m.dtype.names is not None:
            result = self._data.copy('K')
            _recursive_filled(result, self._mask, fill_value)
        elif not m.any():
            return self._data
        else:
            result = self._data.copy('K')
            try:
                np.copyto(result, fill_value, where=m)
            except (TypeError, AttributeError):
                fill_value = narray(fill_value, dtype=object)
                d = result.astype(object)
                result = np.choose(m, (d, fill_value))
            except IndexError:
                if self._data.shape:
                    raise
                elif m:
                    result = np.array(fill_value, dtype=self.dtype)
                else:
                    result = self._data
        return result

    def compressed(self):
        """
        Return all the non-masked data as a 1-D array.

        Returns
        -------
        data : ndarray
            A new `ndarray` holding the non-masked data is returned.

        Notes
        -----
        The result is **not** a MaskedArray!

        Examples
        --------
        >>> x = np.ma.array(np.arange(5), mask=[0]*2 + [1]*3)
        >>> x.compressed()
        array([0, 1])
        >>> type(x.compressed())
        <class 'numpy.ndarray'>

        """
        data = ndarray.ravel(self._data)
        if self._mask is not nomask:
            data = data.compress(np.logical_not(ndarray.ravel(self._mask)))
        return data

    def compress(self, condition, axis=None, out=None):
        """
        Return `a` where condition is ``True``.

        If condition is a `~ma.MaskedArray`, missing values are considered
        as ``False``.

        Parameters
        ----------
        condition : var
            Boolean 1-d array selecting which entries to return. If len(condition)
            is less than the size of a along the axis, then output is truncated
            to length of condition array.
        axis : {None, int}, optional
            Axis along which the operation must be performed.
        out : {None, ndarray}, optional
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary.

        Returns
        -------
        result : MaskedArray
            A :class:`~ma.MaskedArray` object.

        Notes
        -----
        Please note the difference with :meth:`compressed` !
        The output of :meth:`compress` has a mask, the output of
        :meth:`compressed` does not.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.compress([1, 0, 1])
        masked_array(data=[1, 3],
                     mask=[False, False],
               fill_value=999999)

        >>> x.compress([1, 0, 1], axis=1)
        masked_array(
          data=[[1, 3],
                [--, --],
                [7, 9]],
          mask=[[False, False],
                [ True,  True],
                [False, False]],
          fill_value=999999)

        """
        _data, _mask = (self._data, self._mask)
        condition = np.asarray(condition)
        _new = _data.compress(condition, axis=axis, out=out).view(type(self))
        _new._update_from(self)
        if _mask is not nomask:
            _new._mask = _mask.compress(condition, axis=axis)
        return _new

    def _insert_masked_print(self):
        """
        Replace masked values with masked_print_option, casting all innermost
        dtypes to object.
        """
        if masked_print_option.enabled():
            mask = self._mask
            if mask is nomask:
                res = self._data
            else:
                data = self._data
                print_width = self._print_width if self.ndim > 1 else self._print_width_1d
                for axis in range(self.ndim):
                    if data.shape[axis] > print_width:
                        ind = print_width // 2
                        arr = np.split(data, (ind, -ind), axis=axis)
                        data = np.concatenate((arr[0], arr[2]), axis=axis)
                        arr = np.split(mask, (ind, -ind), axis=axis)
                        mask = np.concatenate((arr[0], arr[2]), axis=axis)
                rdtype = _replace_dtype_fields(self.dtype, 'O')
                res = data.astype(rdtype)
                _recursive_printoption(res, mask, masked_print_option)
        else:
            res = self.filled(self.fill_value)
        return res

    def __str__(self):
        return str(self._insert_masked_print())

    def __repr__(self):
        """
        Literal string representation.

        """
        if self._baseclass is np.ndarray:
            name = 'array'
        else:
            name = self._baseclass.__name__
        if np.core.arrayprint._get_legacy_print_mode() <= 113:
            is_long = self.ndim > 1
            parameters = dict(name=name, nlen=' ' * len(name), data=str(self), mask=str(self._mask), fill=str(self.fill_value), dtype=str(self.dtype))
            is_structured = bool(self.dtype.names)
            key = '{}_{}'.format('long' if is_long else 'short', 'flx' if is_structured else 'std')
            return _legacy_print_templates[key] % parameters
        prefix = f'masked_{name}('
        dtype_needed = not np.core.arrayprint.dtype_is_implied(self.dtype) or np.all(self.mask) or self.size == 0
        keys = ['data', 'mask', 'fill_value']
        if dtype_needed:
            keys.append('dtype')
        is_one_row = builtins.all((dim == 1 for dim in self.shape[:-1]))
        min_indent = 2
        if is_one_row:
            indents = {}
            indents[keys[0]] = prefix
            for k in keys[1:]:
                n = builtins.max(min_indent, len(prefix + keys[0]) - len(k))
                indents[k] = ' ' * n
            prefix = ''
        else:
            indents = {k: ' ' * min_indent for k in keys}
            prefix = prefix + '\n'
        reprs = {}
        reprs['data'] = np.array2string(self._insert_masked_print(), separator=', ', prefix=indents['data'] + 'data=', suffix=',')
        reprs['mask'] = np.array2string(self._mask, separator=', ', prefix=indents['mask'] + 'mask=', suffix=',')
        reprs['fill_value'] = repr(self.fill_value)
        if dtype_needed:
            reprs['dtype'] = np.core.arrayprint.dtype_short_repr(self.dtype)
        result = ',\n'.join(('{}{}={}'.format(indents[k], k, reprs[k]) for k in keys))
        return prefix + result + ')'

    def _delegate_binop(self, other):
        if isinstance(other, type(self)):
            return False
        array_ufunc = getattr(other, '__array_ufunc__', False)
        if array_ufunc is False:
            other_priority = getattr(other, '__array_priority__', -1000000)
            return self.__array_priority__ < other_priority
        else:
            return array_ufunc is None

    def _comparison(self, other, compare):
        """Compare self with other using operator.eq or operator.ne.

        When either of the elements is masked, the result is masked as well,
        but the underlying boolean data are still set, with self and other
        considered equal if both are masked, and unequal otherwise.

        For structured arrays, all fields are combined, with masked values
        ignored. The result is masked if all fields were masked, with self
        and other considered equal only if both were fully masked.
        """
        omask = getmask(other)
        smask = self.mask
        mask = mask_or(smask, omask, copy=True)
        odata = getdata(other)
        if mask.dtype.names is not None:
            if compare not in (operator.eq, operator.ne):
                return NotImplemented
            broadcast_shape = np.broadcast(self, odata).shape
            sbroadcast = np.broadcast_to(self, broadcast_shape, subok=True)
            sbroadcast._mask = mask
            sdata = sbroadcast.filled(odata)
            mask = mask == np.ones((), mask.dtype)
            if omask is np.False_:
                omask = np.zeros((), smask.dtype)
        else:
            sdata = self.data
        check = compare(sdata, odata)
        if isinstance(check, (np.bool_, bool)):
            return masked if mask else check
        if mask is not nomask:
            if compare in (operator.eq, operator.ne):
                check = np.where(mask, compare(smask, omask), check)
            if mask.shape != check.shape:
                mask = np.broadcast_to(mask, check.shape).copy()
        check = check.view(type(self))
        check._update_from(self)
        check._mask = mask
        if check._fill_value is not None:
            try:
                fill = _check_fill_value(check._fill_value, np.bool_)
            except (TypeError, ValueError):
                fill = _check_fill_value(None, np.bool_)
            check._fill_value = fill
        return check

    def __eq__(self, other):
        """Check whether other equals self elementwise.

        When either of the elements is masked, the result is masked as well,
        but the underlying boolean data are still set, with self and other
        considered equal if both are masked, and unequal otherwise.

        For structured arrays, all fields are combined, with masked values
        ignored. The result is masked if all fields were masked, with self
        and other considered equal only if both were fully masked.
        """
        return self._comparison(other, operator.eq)

    def __ne__(self, other):
        """Check whether other does not equal self elementwise.

        When either of the elements is masked, the result is masked as well,
        but the underlying boolean data are still set, with self and other
        considered equal if both are masked, and unequal otherwise.

        For structured arrays, all fields are combined, with masked values
        ignored. The result is masked if all fields were masked, with self
        and other considered equal only if both were fully masked.
        """
        return self._comparison(other, operator.ne)

    def __le__(self, other):
        return self._comparison(other, operator.le)

    def __lt__(self, other):
        return self._comparison(other, operator.lt)

    def __ge__(self, other):
        return self._comparison(other, operator.ge)

    def __gt__(self, other):
        return self._comparison(other, operator.gt)

    def __add__(self, other):
        """
        Add self to other, and return a new masked array.

        """
        if self._delegate_binop(other):
            return NotImplemented
        return add(self, other)

    def __radd__(self, other):
        """
        Add other to self, and return a new masked array.

        """
        return add(other, self)

    def __sub__(self, other):
        """
        Subtract other from self, and return a new masked array.

        """
        if self._delegate_binop(other):
            return NotImplemented
        return subtract(self, other)

    def __rsub__(self, other):
        """
        Subtract self from other, and return a new masked array.

        """
        return subtract(other, self)

    def __mul__(self, other):
        """Multiply self by other, and return a new masked array."""
        if self._delegate_binop(other):
            return NotImplemented
        return multiply(self, other)

    def __rmul__(self, other):
        """
        Multiply other by self, and return a new masked array.

        """
        return multiply(other, self)

    def __div__(self, other):
        """
        Divide other into self, and return a new masked array.

        """
        if self._delegate_binop(other):
            return NotImplemented
        return divide(self, other)

    def __truediv__(self, other):
        """
        Divide other into self, and return a new masked array.

        """
        if self._delegate_binop(other):
            return NotImplemented
        return true_divide(self, other)

    def __rtruediv__(self, other):
        """
        Divide self into other, and return a new masked array.

        """
        return true_divide(other, self)

    def __floordiv__(self, other):
        """
        Divide other into self, and return a new masked array.

        """
        if self._delegate_binop(other):
            return NotImplemented
        return floor_divide(self, other)

    def __rfloordiv__(self, other):
        """
        Divide self into other, and return a new masked array.

        """
        return floor_divide(other, self)

    def __pow__(self, other):
        """
        Raise self to the power other, masking the potential NaNs/Infs

        """
        if self._delegate_binop(other):
            return NotImplemented
        return power(self, other)

    def __rpow__(self, other):
        """
        Raise other to the power self, masking the potential NaNs/Infs

        """
        return power(other, self)

    def __iadd__(self, other):
        """
        Add other to self in-place.

        """
        m = getmask(other)
        if self._mask is nomask:
            if m is not nomask and m.any():
                self._mask = make_mask_none(self.shape, self.dtype)
                self._mask += m
        elif m is not nomask:
            self._mask += m
        other_data = getdata(other)
        other_data = np.where(self._mask, other_data.dtype.type(0), other_data)
        self._data.__iadd__(other_data)
        return self

    def __isub__(self, other):
        """
        Subtract other from self in-place.

        """
        m = getmask(other)
        if self._mask is nomask:
            if m is not nomask and m.any():
                self._mask = make_mask_none(self.shape, self.dtype)
                self._mask += m
        elif m is not nomask:
            self._mask += m
        other_data = getdata(other)
        other_data = np.where(self._mask, other_data.dtype.type(0), other_data)
        self._data.__isub__(other_data)
        return self

    def __imul__(self, other):
        """
        Multiply self by other in-place.

        """
        m = getmask(other)
        if self._mask is nomask:
            if m is not nomask and m.any():
                self._mask = make_mask_none(self.shape, self.dtype)
                self._mask += m
        elif m is not nomask:
            self._mask += m
        other_data = getdata(other)
        other_data = np.where(self._mask, other_data.dtype.type(1), other_data)
        self._data.__imul__(other_data)
        return self

    def __idiv__(self, other):
        """
        Divide self by other in-place.

        """
        other_data = getdata(other)
        dom_mask = _DomainSafeDivide().__call__(self._data, other_data)
        other_mask = getmask(other)
        new_mask = mask_or(other_mask, dom_mask)
        if dom_mask.any():
            _, fval = ufunc_fills[np.divide]
            other_data = np.where(dom_mask, other_data.dtype.type(fval), other_data)
        self._mask |= new_mask
        other_data = np.where(self._mask, other_data.dtype.type(1), other_data)
        self._data.__idiv__(other_data)
        return self

    def __ifloordiv__(self, other):
        """
        Floor divide self by other in-place.

        """
        other_data = getdata(other)
        dom_mask = _DomainSafeDivide().__call__(self._data, other_data)
        other_mask = getmask(other)
        new_mask = mask_or(other_mask, dom_mask)
        if dom_mask.any():
            _, fval = ufunc_fills[np.floor_divide]
            other_data = np.where(dom_mask, other_data.dtype.type(fval), other_data)
        self._mask |= new_mask
        other_data = np.where(self._mask, other_data.dtype.type(1), other_data)
        self._data.__ifloordiv__(other_data)
        return self

    def __itruediv__(self, other):
        """
        True divide self by other in-place.

        """
        other_data = getdata(other)
        dom_mask = _DomainSafeDivide().__call__(self._data, other_data)
        other_mask = getmask(other)
        new_mask = mask_or(other_mask, dom_mask)
        if dom_mask.any():
            _, fval = ufunc_fills[np.true_divide]
            other_data = np.where(dom_mask, other_data.dtype.type(fval), other_data)
        self._mask |= new_mask
        other_data = np.where(self._mask, other_data.dtype.type(1), other_data)
        self._data.__itruediv__(other_data)
        return self

    def __ipow__(self, other):
        """
        Raise self to the power other, in place.

        """
        other_data = getdata(other)
        other_data = np.where(self._mask, other_data.dtype.type(1), other_data)
        other_mask = getmask(other)
        with np.errstate(divide='ignore', invalid='ignore'):
            self._data.__ipow__(other_data)
        invalid = np.logical_not(np.isfinite(self._data))
        if invalid.any():
            if self._mask is not nomask:
                self._mask |= invalid
            else:
                self._mask = invalid
            np.copyto(self._data, self.fill_value, where=invalid)
        new_mask = mask_or(other_mask, invalid)
        self._mask = mask_or(self._mask, new_mask)
        return self

    def __float__(self):
        """
        Convert to float.

        """
        if self.size > 1:
            raise TypeError('Only length-1 arrays can be converted to Python scalars')
        elif self._mask:
            warnings.warn('Warning: converting a masked element to nan.', stacklevel=2)
            return np.nan
        return float(self.item())

    def __int__(self):
        """
        Convert to int.

        """
        if self.size > 1:
            raise TypeError('Only length-1 arrays can be converted to Python scalars')
        elif self._mask:
            raise MaskError('Cannot convert masked element to a Python int.')
        return int(self.item())

    @property
    def imag(self):
        """
        The imaginary part of the masked array.

        This property is a view on the imaginary part of this `MaskedArray`.

        See Also
        --------
        real

        Examples
        --------
        >>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
        >>> x.imag
        masked_array(data=[1.0, --, 1.6],
                     mask=[False,  True, False],
               fill_value=1e+20)

        """
        result = self._data.imag.view(type(self))
        result.__setmask__(self._mask)
        return result
    get_imag = imag.fget

    @property
    def real(self):
        """
        The real part of the masked array.

        This property is a view on the real part of this `MaskedArray`.

        See Also
        --------
        imag

        Examples
        --------
        >>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
        >>> x.real
        masked_array(data=[1.0, --, 3.45],
                     mask=[False,  True, False],
               fill_value=1e+20)

        """
        result = self._data.real.view(type(self))
        result.__setmask__(self._mask)
        return result
    get_real = real.fget

    def count(self, axis=None, keepdims=np._NoValue):
        """
        Count the non-masked elements of the array along the given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which the count is performed.
            The default, None, performs the count over all
            the dimensions of the input array. `axis` may be negative, in
            which case it counts from the last to the first axis.

            .. versionadded:: 1.10.0

            If this is a tuple of ints, the count is performed on multiple
            axes, instead of a single axis or all the axes as before.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the array.

        Returns
        -------
        result : ndarray or scalar
            An array with the same shape as the input array, with the specified
            axis removed. If the array is a 0-d array, or if `axis` is None, a
            scalar is returned.

        See Also
        --------
        ma.count_masked : Count masked elements in array or along a given axis.

        Examples
        --------
        >>> import numpy.ma as ma
        >>> a = ma.arange(6).reshape((2, 3))
        >>> a[1, :] = ma.masked
        >>> a
        masked_array(
          data=[[0, 1, 2],
                [--, --, --]],
          mask=[[False, False, False],
                [ True,  True,  True]],
          fill_value=999999)
        >>> a.count()
        3

        When the `axis` keyword is specified an array of appropriate size is
        returned.

        >>> a.count(axis=0)
        array([1, 1, 1])
        >>> a.count(axis=1)
        array([3, 0])

        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
        m = self._mask
        if isinstance(self.data, np.matrix):
            if m is nomask:
                m = np.zeros(self.shape, dtype=np.bool_)
            m = m.view(type(self.data))
        if m is nomask:
            if self.shape == ():
                if axis not in (None, 0):
                    raise np.AxisError(axis=axis, ndim=self.ndim)
                return 1
            elif axis is None:
                if kwargs.get('keepdims', False):
                    return np.array(self.size, dtype=np.intp, ndmin=self.ndim)
                return self.size
            axes = normalize_axis_tuple(axis, self.ndim)
            items = 1
            for ax in axes:
                items *= self.shape[ax]
            if kwargs.get('keepdims', False):
                out_dims = list(self.shape)
                for a in axes:
                    out_dims[a] = 1
            else:
                out_dims = [d for n, d in enumerate(self.shape) if n not in axes]
            return np.full(out_dims, items, dtype=np.intp)
        if self is masked:
            return 0
        return (~m).sum(axis=axis, dtype=np.intp, **kwargs)

    def ravel(self, order='C'):
        """
        Returns a 1D version of self, as a view.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional
            The elements of `a` are read using this index order. 'C' means to
            index the elements in C-like order, with the last axis index
            changing fastest, back to the first axis index changing slowest.
            'F' means to index the elements in Fortran-like index order, with
            the first index changing fastest, and the last index changing
            slowest. Note that the 'C' and 'F' options take no account of the
            memory layout of the underlying array, and only refer to the order
            of axis indexing.  'A' means to read the elements in Fortran-like
            index order if `m` is Fortran *contiguous* in memory, C-like order
            otherwise.  'K' means to read the elements in the order they occur
            in memory, except for reversing the data when strides are negative.
            By default, 'C' index order is used.
            (Masked arrays currently use 'A' on the data when 'K' is passed.)

        Returns
        -------
        MaskedArray
            Output view is of shape ``(self.size,)`` (or
            ``(np.ma.product(self.shape),)``).

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.ravel()
        masked_array(data=[1, --, 3, --, 5, --, 7, --, 9],
                     mask=[False,  True, False,  True, False,  True, False,  True,
                           False],
               fill_value=999999)

        """
        if order in 'kKaA':
            order = 'F' if self._data.flags.fnc else 'C'
        r = ndarray.ravel(self._data, order=order).view(type(self))
        r._update_from(self)
        if self._mask is not nomask:
            r._mask = ndarray.ravel(self._mask, order=order).reshape(r.shape)
        else:
            r._mask = nomask
        return r

    def reshape(self, *s, **kwargs):
        """
        Give a new shape to the array without changing its data.

        Returns a masked array containing the same data, but with a new shape.
        The result is a view on the original array; if this is not possible, a
        ValueError is raised.

        Parameters
        ----------
        shape : int or tuple of ints
            The new shape should be compatible with the original shape. If an
            integer is supplied, then the result will be a 1-D array of that
            length.
        order : {'C', 'F'}, optional
            Determines whether the array data should be viewed as in C
            (row-major) or FORTRAN (column-major) order.

        Returns
        -------
        reshaped_array : array
            A new view on the array.

        See Also
        --------
        reshape : Equivalent function in the masked array module.
        numpy.ndarray.reshape : Equivalent method on ndarray object.
        numpy.reshape : Equivalent function in the NumPy module.

        Notes
        -----
        The reshaping operation cannot guarantee that a copy will not be made,
        to modify the shape in place, use ``a.shape = s``

        Examples
        --------
        >>> x = np.ma.array([[1,2],[3,4]], mask=[1,0,0,1])
        >>> x
        masked_array(
          data=[[--, 2],
                [3, --]],
          mask=[[ True, False],
                [False,  True]],
          fill_value=999999)
        >>> x = x.reshape((4,1))
        >>> x
        masked_array(
          data=[[--],
                [2],
                [3],
                [--]],
          mask=[[ True],
                [False],
                [False],
                [ True]],
          fill_value=999999)

        """
        kwargs.update(order=kwargs.get('order', 'C'))
        result = self._data.reshape(*s, **kwargs).view(type(self))
        result._update_from(self)
        mask = self._mask
        if mask is not nomask:
            result._mask = mask.reshape(*s, **kwargs)
        return result

    def resize(self, newshape, refcheck=True, order=False):
        """
        .. warning::

            This method does nothing, except raise a ValueError exception. A
            masked array does not own its data and therefore cannot safely be
            resized in place. Use the `numpy.ma.resize` function instead.

        This method is difficult to implement safely and may be deprecated in
        future releases of NumPy.

        """
        errmsg = 'A masked array does not own its data and therefore cannot be resized.\nUse the numpy.ma.resize function instead.'
        raise ValueError(errmsg)

    def put(self, indices, values, mode='raise'):
        """
        Set storage-indexed locations to corresponding values.

        Sets self._data.flat[n] = values[n] for each n in indices.
        If `values` is shorter than `indices` then it will repeat.
        If `values` has some masked values, the initial mask is updated
        in consequence, else the corresponding values are unmasked.

        Parameters
        ----------
        indices : 1-D array_like
            Target indices, interpreted as integers.
        values : array_like
            Values to place in self._data copy at target indices.
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how out-of-bounds indices will behave.
            'raise' : raise an error.
            'wrap' : wrap around.
            'clip' : clip to the range.

        Notes
        -----
        `values` can be a scalar or length 1 array.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.put([0,4,8],[10,20,30])
        >>> x
        masked_array(
          data=[[10, --, 3],
                [--, 20, --],
                [7, --, 30]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)

        >>> x.put(4,999)
        >>> x
        masked_array(
          data=[[10, --, 3],
                [--, 999, --],
                [7, --, 30]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)

        """
        if self._hardmask and self._mask is not nomask:
            mask = self._mask[indices]
            indices = narray(indices, copy=False)
            values = narray(values, copy=False, subok=True)
            values.resize(indices.shape)
            indices = indices[~mask]
            values = values[~mask]
        self._data.put(indices, values, mode=mode)
        if self._mask is nomask and getmask(values) is nomask:
            return
        m = getmaskarray(self)
        if getmask(values) is nomask:
            m.put(indices, False, mode=mode)
        else:
            m.put(indices, values._mask, mode=mode)
        m = make_mask(m, copy=False, shrink=True)
        self._mask = m
        return

    def ids(self):
        """
        Return the addresses of the data and mask areas.

        Parameters
        ----------
        None

        Examples
        --------
        >>> x = np.ma.array([1, 2, 3], mask=[0, 1, 1])
        >>> x.ids()
        (166670640, 166659832) # may vary

        If the array has no mask, the address of `nomask` is returned. This address
        is typically not close to the data in memory:

        >>> x = np.ma.array([1, 2, 3])
        >>> x.ids()
        (166691080, 3083169284) # may vary

        """
        if self._mask is nomask:
            return (self.ctypes.data, id(nomask))
        return (self.ctypes.data, self._mask.ctypes.data)

    def iscontiguous(self):
        """
        Return a boolean indicating whether the data is contiguous.

        Parameters
        ----------
        None

        Examples
        --------
        >>> x = np.ma.array([1, 2, 3])
        >>> x.iscontiguous()
        True

        `iscontiguous` returns one of the flags of the masked array:

        >>> x.flags
          C_CONTIGUOUS : True
          F_CONTIGUOUS : True
          OWNDATA : False
          WRITEABLE : True
          ALIGNED : True
          WRITEBACKIFCOPY : False

        """
        return self.flags['CONTIGUOUS']

    def all(self, axis=None, out=None, keepdims=np._NoValue):
        """
        Returns True if all elements evaluate to True.

        The output array is masked where all the values along the given axis
        are masked: if the output would have been a scalar and that all the
        values are masked, then the output is `masked`.

        Refer to `numpy.all` for full documentation.

        See Also
        --------
        numpy.ndarray.all : corresponding function for ndarrays
        numpy.all : equivalent function

        Examples
        --------
        >>> np.ma.array([1,2,3]).all()
        True
        >>> a = np.ma.array([1,2,3], mask=True)
        >>> (a.all() is np.ma.masked)
        True

        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
        mask = _check_mask_axis(self._mask, axis, **kwargs)
        if out is None:
            d = self.filled(True).all(axis=axis, **kwargs).view(type(self))
            if d.ndim:
                d.__setmask__(mask)
            elif mask:
                return masked
            return d
        self.filled(True).all(axis=axis, out=out, **kwargs)
        if isinstance(out, MaskedArray):
            if out.ndim or mask:
                out.__setmask__(mask)
        return out

    def any(self, axis=None, out=None, keepdims=np._NoValue):
        """
        Returns True if any of the elements of `a` evaluate to True.

        Masked values are considered as False during computation.

        Refer to `numpy.any` for full documentation.

        See Also
        --------
        numpy.ndarray.any : corresponding function for ndarrays
        numpy.any : equivalent function

        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
        mask = _check_mask_axis(self._mask, axis, **kwargs)
        if out is None:
            d = self.filled(False).any(axis=axis, **kwargs).view(type(self))
            if d.ndim:
                d.__setmask__(mask)
            elif mask:
                d = masked
            return d
        self.filled(False).any(axis=axis, out=out, **kwargs)
        if isinstance(out, MaskedArray):
            if out.ndim or mask:
                out.__setmask__(mask)
        return out

    def nonzero(self):
        """
        Return the indices of unmasked elements that are not zero.

        Returns a tuple of arrays, one for each dimension, containing the
        indices of the non-zero elements in that dimension. The corresponding
        non-zero values can be obtained with::

            a[a.nonzero()]

        To group the indices by element, rather than dimension, use
        instead::

            np.transpose(a.nonzero())

        The result of this is always a 2d array, with a row for each non-zero
        element.

        Parameters
        ----------
        None

        Returns
        -------
        tuple_of_arrays : tuple
            Indices of elements that are non-zero.

        See Also
        --------
        numpy.nonzero :
            Function operating on ndarrays.
        flatnonzero :
            Return indices that are non-zero in the flattened version of the input
            array.
        numpy.ndarray.nonzero :
            Equivalent ndarray method.
        count_nonzero :
            Counts the number of non-zero elements in the input array.

        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = ma.array(np.eye(3))
        >>> x
        masked_array(
          data=[[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]],
          mask=False,
          fill_value=1e+20)
        >>> x.nonzero()
        (array([0, 1, 2]), array([0, 1, 2]))

        Masked elements are ignored.

        >>> x[1, 1] = ma.masked
        >>> x
        masked_array(
          data=[[1.0, 0.0, 0.0],
                [0.0, --, 0.0],
                [0.0, 0.0, 1.0]],
          mask=[[False, False, False],
                [False,  True, False],
                [False, False, False]],
          fill_value=1e+20)
        >>> x.nonzero()
        (array([0, 2]), array([0, 2]))

        Indices can also be grouped by element.

        >>> np.transpose(x.nonzero())
        array([[0, 0],
               [2, 2]])

        A common use for ``nonzero`` is to find the indices of an array, where
        a condition is True.  Given an array `a`, the condition `a` > 3 is a
        boolean array and since False is interpreted as 0, ma.nonzero(a > 3)
        yields the indices of the `a` where the condition is true.

        >>> a = ma.array([[1,2,3],[4,5,6],[7,8,9]])
        >>> a > 3
        masked_array(
          data=[[False, False, False],
                [ True,  True,  True],
                [ True,  True,  True]],
          mask=False,
          fill_value=True)
        >>> ma.nonzero(a > 3)
        (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

        The ``nonzero`` method of the condition array can also be called.

        >>> (a > 3).nonzero()
        (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

        """
        return narray(self.filled(0), copy=False).nonzero()

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """
        (this docstring should be overwritten)
        """
        m = self._mask
        if m is nomask:
            result = super().trace(offset=offset, axis1=axis1, axis2=axis2, out=out)
            return result.astype(dtype)
        else:
            D = self.diagonal(offset=offset, axis1=axis1, axis2=axis2)
            return D.astype(dtype).filled(0).sum(axis=-1, out=out)
    trace.__doc__ = ndarray.trace.__doc__

    def dot(self, b, out=None, strict=False):
        """
        a.dot(b, out=None)

        Masked dot product of two arrays. Note that `out` and `strict` are
        located in different positions than in `ma.dot`. In order to
        maintain compatibility with the functional version, it is
        recommended that the optional arguments be treated as keyword only.
        At some point that may be mandatory.

        .. versionadded:: 1.10.0

        Parameters
        ----------
        b : masked_array_like
            Inputs array.
        out : masked_array, optional
            Output argument. This must have the exact kind that would be
            returned if it was not used. In particular, it must have the
            right type, must be C-contiguous, and its dtype must be the
            dtype that would be returned for `ma.dot(a,b)`. This is a
            performance feature. Therefore, if these conditions are not
            met, an exception is raised, instead of attempting to be
            flexible.
        strict : bool, optional
            Whether masked data are propagated (True) or set to 0 (False)
            for the computation. Default is False.  Propagating the mask
            means that if a masked value appears in a row or column, the
            whole row or column is considered masked.

            .. versionadded:: 1.10.2

        See Also
        --------
        numpy.ma.dot : equivalent function

        """
        return dot(self, b, out=out, strict=strict)

    def sum(self, axis=None, dtype=None, out=None, keepdims=np._NoValue):
        """
        Return the sum of the array elements over the given axis.

        Masked elements are set to 0 internally.

        Refer to `numpy.sum` for full documentation.

        See Also
        --------
        numpy.ndarray.sum : corresponding function for ndarrays
        numpy.sum : equivalent function

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.sum()
        25
        >>> x.sum(axis=1)
        masked_array(data=[4, 5, 16],
                     mask=[False, False, False],
               fill_value=999999)
        >>> x.sum(axis=0)
        masked_array(data=[8, 5, 12],
                     mask=[False, False, False],
               fill_value=999999)
        >>> print(type(x.sum(axis=0, dtype=np.int64)[0]))
        <class 'numpy.int64'>

        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
        _mask = self._mask
        newmask = _check_mask_axis(_mask, axis, **kwargs)
        if out is None:
            result = self.filled(0).sum(axis, dtype=dtype, **kwargs)
            rndim = getattr(result, 'ndim', 0)
            if rndim:
                result = result.view(type(self))
                result.__setmask__(newmask)
            elif newmask:
                result = masked
            return result
        result = self.filled(0).sum(axis, dtype=dtype, out=out, **kwargs)
        if isinstance(out, MaskedArray):
            outmask = getmask(out)
            if outmask is nomask:
                outmask = out._mask = make_mask_none(out.shape)
            outmask.flat = newmask
        return out

    def cumsum(self, axis=None, dtype=None, out=None):
        """
        Return the cumulative sum of the array elements over the given axis.

        Masked values are set to 0 internally during the computation.
        However, their position is saved, and the result will be masked at
        the same locations.

        Refer to `numpy.cumsum` for full documentation.

        Notes
        -----
        The mask is lost if `out` is not a valid :class:`ma.MaskedArray` !

        Arithmetic is modular when using integer types, and no error is
        raised on overflow.

        See Also
        --------
        numpy.ndarray.cumsum : corresponding function for ndarrays
        numpy.cumsum : equivalent function

        Examples
        --------
        >>> marr = np.ma.array(np.arange(10), mask=[0,0,0,1,1,1,0,0,0,0])
        >>> marr.cumsum()
        masked_array(data=[0, 1, 3, --, --, --, 9, 16, 24, 33],
                     mask=[False, False, False,  True,  True,  True, False, False,
                           False, False],
               fill_value=999999)

        """
        result = self.filled(0).cumsum(axis=axis, dtype=dtype, out=out)
        if out is not None:
            if isinstance(out, MaskedArray):
                out.__setmask__(self.mask)
            return out
        result = result.view(type(self))
        result.__setmask__(self._mask)
        return result

    def prod(self, axis=None, dtype=None, out=None, keepdims=np._NoValue):
        """
        Return the product of the array elements over the given axis.

        Masked elements are set to 1 internally for computation.

        Refer to `numpy.prod` for full documentation.

        Notes
        -----
        Arithmetic is modular when using integer types, and no error is raised
        on overflow.

        See Also
        --------
        numpy.ndarray.prod : corresponding function for ndarrays
        numpy.prod : equivalent function
        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
        _mask = self._mask
        newmask = _check_mask_axis(_mask, axis, **kwargs)
        if out is None:
            result = self.filled(1).prod(axis, dtype=dtype, **kwargs)
            rndim = getattr(result, 'ndim', 0)
            if rndim:
                result = result.view(type(self))
                result.__setmask__(newmask)
            elif newmask:
                result = masked
            return result
        result = self.filled(1).prod(axis, dtype=dtype, out=out, **kwargs)
        if isinstance(out, MaskedArray):
            outmask = getmask(out)
            if outmask is nomask:
                outmask = out._mask = make_mask_none(out.shape)
            outmask.flat = newmask
        return out
    product = prod

    def cumprod(self, axis=None, dtype=None, out=None):
        """
        Return the cumulative product of the array elements over the given axis.

        Masked values are set to 1 internally during the computation.
        However, their position is saved, and the result will be masked at
        the same locations.

        Refer to `numpy.cumprod` for full documentation.

        Notes
        -----
        The mask is lost if `out` is not a valid MaskedArray !

        Arithmetic is modular when using integer types, and no error is
        raised on overflow.

        See Also
        --------
        numpy.ndarray.cumprod : corresponding function for ndarrays
        numpy.cumprod : equivalent function
        """
        result = self.filled(1).cumprod(axis=axis, dtype=dtype, out=out)
        if out is not None:
            if isinstance(out, MaskedArray):
                out.__setmask__(self._mask)
            return out
        result = result.view(type(self))
        result.__setmask__(self._mask)
        return result

    def mean(self, axis=None, dtype=None, out=None, keepdims=np._NoValue):
        """
        Returns the average of the array elements along given axis.

        Masked entries are ignored, and result elements which are not
        finite will be masked.

        Refer to `numpy.mean` for full documentation.

        See Also
        --------
        numpy.ndarray.mean : corresponding function for ndarrays
        numpy.mean : Equivalent function
        numpy.ma.average : Weighted average.

        Examples
        --------
        >>> a = np.ma.array([1,2,3], mask=[False, False, True])
        >>> a
        masked_array(data=[1, 2, --],
                     mask=[False, False,  True],
               fill_value=999999)
        >>> a.mean()
        1.5

        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
        if self._mask is nomask:
            result = super().mean(axis=axis, dtype=dtype, **kwargs)[()]
        else:
            is_float16_result = False
            if dtype is None:
                if issubclass(self.dtype.type, (ntypes.integer, ntypes.bool_)):
                    dtype = mu.dtype('f8')
                elif issubclass(self.dtype.type, ntypes.float16):
                    dtype = mu.dtype('f4')
                    is_float16_result = True
            dsum = self.sum(axis=axis, dtype=dtype, **kwargs)
            cnt = self.count(axis=axis, **kwargs)
            if cnt.shape == () and cnt == 0:
                result = masked
            elif is_float16_result:
                result = self.dtype.type(dsum * 1.0 / cnt)
            else:
                result = dsum * 1.0 / cnt
        if out is not None:
            out.flat = result
            if isinstance(out, MaskedArray):
                outmask = getmask(out)
                if outmask is nomask:
                    outmask = out._mask = make_mask_none(out.shape)
                outmask.flat = getmask(result)
            return out
        return result

    def anom(self, axis=None, dtype=None):
        """
        Compute the anomalies (deviations from the arithmetic mean)
        along the given axis.

        Returns an array of anomalies, with the same shape as the input and
        where the arithmetic mean is computed along the given axis.

        Parameters
        ----------
        axis : int, optional
            Axis over which the anomalies are taken.
            The default is to use the mean of the flattened array as reference.
        dtype : dtype, optional
            Type to use in computing the variance. For arrays of integer type
             the default is float32; for arrays of float types it is the same as
             the array type.

        See Also
        --------
        mean : Compute the mean of the array.

        Examples
        --------
        >>> a = np.ma.array([1,2,3])
        >>> a.anom()
        masked_array(data=[-1.,  0.,  1.],
                     mask=False,
               fill_value=1e+20)

        """
        m = self.mean(axis, dtype)
        if not axis:
            return self - m
        else:
            return self - expand_dims(m, axis)

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
        """
        Returns the variance of the array elements along given axis.

        Masked entries are ignored, and result elements which are not
        finite will be masked.

        Refer to `numpy.var` for full documentation.

        See Also
        --------
        numpy.ndarray.var : corresponding function for ndarrays
        numpy.var : Equivalent function
        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
        if self._mask is nomask:
            ret = super().var(axis=axis, dtype=dtype, out=out, ddof=ddof, **kwargs)[()]
            if out is not None:
                if isinstance(out, MaskedArray):
                    out.__setmask__(nomask)
                return out
            return ret
        cnt = self.count(axis=axis, **kwargs) - ddof
        danom = self - self.mean(axis, dtype, keepdims=True)
        if iscomplexobj(self):
            danom = umath.absolute(danom) ** 2
        else:
            danom *= danom
        dvar = divide(danom.sum(axis, **kwargs), cnt).view(type(self))
        if dvar.ndim:
            dvar._mask = mask_or(self._mask.all(axis, **kwargs), cnt <= 0)
            dvar._update_from(self)
        elif getmask(dvar):
            dvar = masked
            if out is not None:
                if isinstance(out, MaskedArray):
                    out.flat = 0
                    out.__setmask__(True)
                elif out.dtype.kind in 'biu':
                    errmsg = 'Masked data information would be lost in one or more location.'
                    raise MaskError(errmsg)
                else:
                    out.flat = np.nan
                return out
        if out is not None:
            out.flat = dvar
            if isinstance(out, MaskedArray):
                out.__setmask__(dvar.mask)
            return out
        return dvar
    var.__doc__ = np.var.__doc__

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
        """
        Returns the standard deviation of the array elements along given axis.

        Masked entries are ignored.

        Refer to `numpy.std` for full documentation.

        See Also
        --------
        numpy.ndarray.std : corresponding function for ndarrays
        numpy.std : Equivalent function
        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
        dvar = self.var(axis, dtype, out, ddof, **kwargs)
        if dvar is not masked:
            if out is not None:
                np.power(out, 0.5, out=out, casting='unsafe')
                return out
            dvar = sqrt(dvar)
        return dvar

    def round(self, decimals=0, out=None):
        """
        Return each element rounded to the given number of decimals.

        Refer to `numpy.around` for full documentation.

        See Also
        --------
        numpy.ndarray.round : corresponding function for ndarrays
        numpy.around : equivalent function
        """
        result = self._data.round(decimals=decimals, out=out).view(type(self))
        if result.ndim > 0:
            result._mask = self._mask
            result._update_from(self)
        elif self._mask:
            result = masked
        if out is None:
            return result
        if isinstance(out, MaskedArray):
            out.__setmask__(self._mask)
        return out

    def argsort(self, axis=np._NoValue, kind=None, order=None, endwith=True, fill_value=None):
        """
        Return an ndarray of indices that sort the array along the
        specified axis.  Masked values are filled beforehand to
        `fill_value`.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. If None, the default, the flattened array
            is used.

            ..  versionchanged:: 1.13.0
                Previously, the default was documented to be -1, but that was
                in error. At some future date, the default will change to -1, as
                originally intended.
                Until then, the axis should be given explicitly when
                ``arr.ndim > 1``, to avoid a FutureWarning.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            The sorting algorithm used.
        order : list, optional
            When `a` is an array with fields defined, this argument specifies
            which fields to compare first, second, etc.  Not all fields need be
            specified.
        endwith : {True, False}, optional
            Whether missing values (if any) should be treated as the largest values
            (True) or the smallest values (False)
            When the array contains unmasked values at the same extremes of the
            datatype, the ordering of these values and the masked values is
            undefined.
        fill_value : scalar or None, optional
            Value used internally for the masked values.
            If ``fill_value`` is not None, it supersedes ``endwith``.

        Returns
        -------
        index_array : ndarray, int
            Array of indices that sort `a` along the specified axis.
            In other words, ``a[index_array]`` yields a sorted `a`.

        See Also
        --------
        ma.MaskedArray.sort : Describes sorting algorithms used.
        lexsort : Indirect stable sort with multiple keys.
        numpy.ndarray.sort : Inplace sort.

        Notes
        -----
        See `sort` for notes on the different sorting algorithms.

        Examples
        --------
        >>> a = np.ma.array([3,2,1], mask=[False, False, True])
        >>> a
        masked_array(data=[3, 2, --],
                     mask=[False, False,  True],
               fill_value=999999)
        >>> a.argsort()
        array([1, 0, 2])

        """
        if axis is np._NoValue:
            axis = _deprecate_argsort_axis(self)
        if fill_value is None:
            if endwith:
                if np.issubdtype(self.dtype, np.floating):
                    fill_value = np.nan
                else:
                    fill_value = minimum_fill_value(self)
            else:
                fill_value = maximum_fill_value(self)
        filled = self.filled(fill_value)
        return filled.argsort(axis=axis, kind=kind, order=order)

    def argmin(self, axis=None, fill_value=None, out=None, *, keepdims=np._NoValue):
        """
        Return array of indices to the minimum values along the given axis.

        Parameters
        ----------
        axis : {None, integer}
            If None, the index is into the flattened array, otherwise along
            the specified axis
        fill_value : scalar or None, optional
            Value used to fill in the masked values.  If None, the output of
            minimum_fill_value(self._data) is used instead.
        out : {None, array}, optional
            Array into which the result can be placed. Its type is preserved
            and it must be of the right shape to hold the output.

        Returns
        -------
        ndarray or scalar
            If multi-dimension input, returns a new ndarray of indices to the
            minimum values along the given axis.  Otherwise, returns a scalar
            of index to the minimum values along the given axis.

        Examples
        --------
        >>> x = np.ma.array(np.arange(4), mask=[1,1,0,0])
        >>> x.shape = (2,2)
        >>> x
        masked_array(
          data=[[--, --],
                [2, 3]],
          mask=[[ True,  True],
                [False, False]],
          fill_value=999999)
        >>> x.argmin(axis=0, fill_value=-1)
        array([0, 0])
        >>> x.argmin(axis=0, fill_value=9)
        array([1, 1])

        """
        if fill_value is None:
            fill_value = minimum_fill_value(self)
        d = self.filled(fill_value).view(ndarray)
        keepdims = False if keepdims is np._NoValue else bool(keepdims)
        return d.argmin(axis, out=out, keepdims=keepdims)

    def argmax(self, axis=None, fill_value=None, out=None, *, keepdims=np._NoValue):
        """
        Returns array of indices of the maximum values along the given axis.
        Masked values are treated as if they had the value fill_value.

        Parameters
        ----------
        axis : {None, integer}
            If None, the index is into the flattened array, otherwise along
            the specified axis
        fill_value : scalar or None, optional
            Value used to fill in the masked values.  If None, the output of
            maximum_fill_value(self._data) is used instead.
        out : {None, array}, optional
            Array into which the result can be placed. Its type is preserved
            and it must be of the right shape to hold the output.

        Returns
        -------
        index_array : {integer_array}

        Examples
        --------
        >>> a = np.arange(6).reshape(2,3)
        >>> a.argmax()
        5
        >>> a.argmax(0)
        array([1, 1, 1])
        >>> a.argmax(1)
        array([2, 2])

        """
        if fill_value is None:
            fill_value = maximum_fill_value(self._data)
        d = self.filled(fill_value).view(ndarray)
        keepdims = False if keepdims is np._NoValue else bool(keepdims)
        return d.argmax(axis, out=out, keepdims=keepdims)

    def sort(self, axis=-1, kind=None, order=None, endwith=True, fill_value=None):
        """
        Sort the array, in-place

        Parameters
        ----------
        a : array_like
            Array to be sorted.
        axis : int, optional
            Axis along which to sort. If None, the array is flattened before
            sorting. The default is -1, which sorts along the last axis.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            The sorting algorithm used.
        order : list, optional
            When `a` is a structured array, this argument specifies which fields
            to compare first, second, and so on.  This list does not need to
            include all of the fields.
        endwith : {True, False}, optional
            Whether missing values (if any) should be treated as the largest values
            (True) or the smallest values (False)
            When the array contains unmasked values sorting at the same extremes of the
            datatype, the ordering of these values and the masked values is
            undefined.
        fill_value : scalar or None, optional
            Value used internally for the masked values.
            If ``fill_value`` is not None, it supersedes ``endwith``.

        Returns
        -------
        sorted_array : ndarray
            Array of the same type and shape as `a`.

        See Also
        --------
        numpy.ndarray.sort : Method to sort an array in-place.
        argsort : Indirect sort.
        lexsort : Indirect stable sort on multiple keys.
        searchsorted : Find elements in a sorted array.

        Notes
        -----
        See ``sort`` for notes on the different sorting algorithms.

        Examples
        --------
        >>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
        >>> # Default
        >>> a.sort()
        >>> a
        masked_array(data=[1, 3, 5, --, --],
                     mask=[False, False, False,  True,  True],
               fill_value=999999)

        >>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
        >>> # Put missing values in the front
        >>> a.sort(endwith=False)
        >>> a
        masked_array(data=[--, --, 1, 3, 5],
                     mask=[ True,  True, False, False, False],
               fill_value=999999)

        >>> a = np.ma.array([1, 2, 5, 4, 3],mask=[0, 1, 0, 1, 0])
        >>> # fill_value takes over endwith
        >>> a.sort(endwith=False, fill_value=3)
        >>> a
        masked_array(data=[1, --, --, 3, 5],
                     mask=[False,  True,  True, False, False],
               fill_value=999999)

        """
        if self._mask is nomask:
            ndarray.sort(self, axis=axis, kind=kind, order=order)
            return
        if self is masked:
            return
        sidx = self.argsort(axis=axis, kind=kind, order=order, fill_value=fill_value, endwith=endwith)
        self[...] = np.take_along_axis(self, sidx, axis=axis)

    def min(self, axis=None, out=None, fill_value=None, keepdims=np._NoValue):
        """
        Return the minimum along a given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis along which to operate.  By default, ``axis`` is None and the
            flattened input is used.
            .. versionadded:: 1.7.0
            If this is a tuple of ints, the minimum is selected over multiple
            axes, instead of a single axis or all the axes as before.
        out : array_like, optional
            Alternative output array in which to place the result.  Must be of
            the same shape and buffer length as the expected output.
        fill_value : scalar or None, optional
            Value used to fill in the masked values.
            If None, use the output of `minimum_fill_value`.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the array.

        Returns
        -------
        amin : array_like
            New array holding the result.
            If ``out`` was specified, ``out`` is returned.

        See Also
        --------
        ma.minimum_fill_value
            Returns the minimum filling value for a given datatype.

        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = [[1., -2., 3.], [0.2, -0.7, 0.1]]
        >>> mask = [[1, 1, 0], [0, 0, 1]]
        >>> masked_x = ma.masked_array(x, mask)
        >>> masked_x
        masked_array(
          data=[[--, --, 3.0],
                [0.2, -0.7, --]],
          mask=[[ True,  True, False],
                [False, False,  True]],
          fill_value=1e+20)
        >>> ma.min(masked_x)
        -0.7
        >>> ma.min(masked_x, axis=-1)
        masked_array(data=[3.0, -0.7],
                     mask=[False, False],
                fill_value=1e+20)
        >>> ma.min(masked_x, axis=0, keepdims=True)
        masked_array(data=[[0.2, -0.7, 3.0]],
                     mask=[[False, False, False]],
                fill_value=1e+20)
        >>> mask = [[1, 1, 1,], [1, 1, 1]]
        >>> masked_x = ma.masked_array(x, mask)
        >>> ma.min(masked_x, axis=0)
        masked_array(data=[--, --, --],
                     mask=[ True,  True,  True],
                fill_value=1e+20,
                    dtype=float64)
        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
        _mask = self._mask
        newmask = _check_mask_axis(_mask, axis, **kwargs)
        if fill_value is None:
            fill_value = minimum_fill_value(self)
        if out is None:
            result = self.filled(fill_value).min(axis=axis, out=out, **kwargs).view(type(self))
            if result.ndim:
                result.__setmask__(newmask)
                if newmask.ndim:
                    np.copyto(result, result.fill_value, where=newmask)
            elif newmask:
                result = masked
            return result
        result = self.filled(fill_value).min(axis=axis, out=out, **kwargs)
        if isinstance(out, MaskedArray):
            outmask = getmask(out)
            if outmask is nomask:
                outmask = out._mask = make_mask_none(out.shape)
            outmask.flat = newmask
        else:
            if out.dtype.kind in 'biu':
                errmsg = 'Masked data information would be lost in one or more location.'
                raise MaskError(errmsg)
            np.copyto(out, np.nan, where=newmask)
        return out

    def max(self, axis=None, out=None, fill_value=None, keepdims=np._NoValue):
        """
        Return the maximum along a given axis.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis along which to operate.  By default, ``axis`` is None and the
            flattened input is used.
            .. versionadded:: 1.7.0
            If this is a tuple of ints, the maximum is selected over multiple
            axes, instead of a single axis or all the axes as before.
        out : array_like, optional
            Alternative output array in which to place the result.  Must
            be of the same shape and buffer length as the expected output.
        fill_value : scalar or None, optional
            Value used to fill in the masked values.
            If None, use the output of maximum_fill_value().
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the array.

        Returns
        -------
        amax : array_like
            New array holding the result.
            If ``out`` was specified, ``out`` is returned.

        See Also
        --------
        ma.maximum_fill_value
            Returns the maximum filling value for a given datatype.

        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = [[-1., 2.5], [4., -2.], [3., 0.]]
        >>> mask = [[0, 0], [1, 0], [1, 0]]
        >>> masked_x = ma.masked_array(x, mask)
        >>> masked_x
        masked_array(
          data=[[-1.0, 2.5],
                [--, -2.0],
                [--, 0.0]],
          mask=[[False, False],
                [ True, False],
                [ True, False]],
          fill_value=1e+20)
        >>> ma.max(masked_x)
        2.5
        >>> ma.max(masked_x, axis=0)
        masked_array(data=[-1.0, 2.5],
                     mask=[False, False],
               fill_value=1e+20)
        >>> ma.max(masked_x, axis=1, keepdims=True)
        masked_array(
          data=[[2.5],
                [-2.0],
                [0.0]],
          mask=[[False],
                [False],
                [False]],
          fill_value=1e+20)
        >>> mask = [[1, 1], [1, 1], [1, 1]]
        >>> masked_x = ma.masked_array(x, mask)
        >>> ma.max(masked_x, axis=1)
        masked_array(data=[--, --, --],
                     mask=[ True,  True,  True],
               fill_value=1e+20,
                    dtype=float64)
        """
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
        _mask = self._mask
        newmask = _check_mask_axis(_mask, axis, **kwargs)
        if fill_value is None:
            fill_value = maximum_fill_value(self)
        if out is None:
            result = self.filled(fill_value).max(axis=axis, out=out, **kwargs).view(type(self))
            if result.ndim:
                result.__setmask__(newmask)
                if newmask.ndim:
                    np.copyto(result, result.fill_value, where=newmask)
            elif newmask:
                result = masked
            return result
        result = self.filled(fill_value).max(axis=axis, out=out, **kwargs)
        if isinstance(out, MaskedArray):
            outmask = getmask(out)
            if outmask is nomask:
                outmask = out._mask = make_mask_none(out.shape)
            outmask.flat = newmask
        else:
            if out.dtype.kind in 'biu':
                errmsg = 'Masked data information would be lost in one or more location.'
                raise MaskError(errmsg)
            np.copyto(out, np.nan, where=newmask)
        return out

    def ptp(self, axis=None, out=None, fill_value=None, keepdims=False):
        """
        Return (maximum - minimum) along the given dimension
        (i.e. peak-to-peak value).

        .. warning::
            `ptp` preserves the data type of the array. This means the
            return value for an input of signed integers with n bits
            (e.g. `np.int8`, `np.int16`, etc) is also a signed integer
            with n bits.  In that case, peak-to-peak values greater than
            ``2**(n-1)-1`` will be returned as negative values. An example
            with a work-around is shown below.

        Parameters
        ----------
        axis : {None, int}, optional
            Axis along which to find the peaks.  If None (default) the
            flattened array is used.
        out : {None, array_like}, optional
            Alternative output array in which to place the result. It must
            have the same shape and buffer length as the expected output
            but the type will be cast if necessary.
        fill_value : scalar or None, optional
            Value used to fill in the masked values.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the array.

        Returns
        -------
        ptp : ndarray.
            A new array holding the result, unless ``out`` was
            specified, in which case a reference to ``out`` is returned.

        Examples
        --------
        >>> x = np.ma.MaskedArray([[4, 9, 2, 10],
        ...                        [6, 9, 7, 12]])

        >>> x.ptp(axis=1)
        masked_array(data=[8, 6],
                     mask=False,
               fill_value=999999)

        >>> x.ptp(axis=0)
        masked_array(data=[2, 0, 5, 2],
                     mask=False,
               fill_value=999999)

        >>> x.ptp()
        10

        This example shows that a negative value can be returned when
        the input is an array of signed integers.

        >>> y = np.ma.MaskedArray([[1, 127],
        ...                        [0, 127],
        ...                        [-1, 127],
        ...                        [-2, 127]], dtype=np.int8)
        >>> y.ptp(axis=1)
        masked_array(data=[ 126,  127, -128, -127],
                     mask=False,
               fill_value=999999,
                    dtype=int8)

        A work-around is to use the `view()` method to view the result as
        unsigned integers with the same bit width:

        >>> y.ptp(axis=1).view(np.uint8)
        masked_array(data=[126, 127, 128, 129],
                     mask=False,
               fill_value=999999,
                    dtype=uint8)
        """
        if out is None:
            result = self.max(axis=axis, fill_value=fill_value, keepdims=keepdims)
            result -= self.min(axis=axis, fill_value=fill_value, keepdims=keepdims)
            return result
        out.flat = self.max(axis=axis, out=out, fill_value=fill_value, keepdims=keepdims)
        min_value = self.min(axis=axis, fill_value=fill_value, keepdims=keepdims)
        np.subtract(out, min_value, out=out, casting='unsafe')
        return out

    def partition(self, *args, **kwargs):
        warnings.warn(f"Warning: 'partition' will ignore the 'mask' of the {self.__class__.__name__}.", stacklevel=2)
        return super().partition(*args, **kwargs)

    def argpartition(self, *args, **kwargs):
        warnings.warn(f"Warning: 'argpartition' will ignore the 'mask' of the {self.__class__.__name__}.", stacklevel=2)
        return super().argpartition(*args, **kwargs)

    def take(self, indices, axis=None, out=None, mode='raise'):
        """
        """
        _data, _mask = (self._data, self._mask)
        cls = type(self)
        maskindices = getmask(indices)
        if maskindices is not nomask:
            indices = indices.filled(0)
        if out is None:
            out = _data.take(indices, axis=axis, mode=mode)[...].view(cls)
        else:
            np.take(_data, indices, axis=axis, mode=mode, out=out)
        if isinstance(out, MaskedArray):
            if _mask is nomask:
                outmask = maskindices
            else:
                outmask = _mask.take(indices, axis=axis, mode=mode)
                outmask |= maskindices
            out.__setmask__(outmask)
        return out[()]
    copy = _arraymethod('copy')
    diagonal = _arraymethod('diagonal')
    flatten = _arraymethod('flatten')
    repeat = _arraymethod('repeat')
    squeeze = _arraymethod('squeeze')
    swapaxes = _arraymethod('swapaxes')
    T = property(fget=lambda self: self.transpose())
    transpose = _arraymethod('transpose')

    def tolist(self, fill_value=None):
        """
        Return the data portion of the masked array as a hierarchical Python list.

        Data items are converted to the nearest compatible Python type.
        Masked values are converted to `fill_value`. If `fill_value` is None,
        the corresponding entries in the output list will be ``None``.

        Parameters
        ----------
        fill_value : scalar, optional
            The value to use for invalid entries. Default is None.

        Returns
        -------
        result : list
            The Python list representation of the masked array.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3], [4,5,6], [7,8,9]], mask=[0] + [1,0]*4)
        >>> x.tolist()
        [[1, None, 3], [None, 5, None], [7, None, 9]]
        >>> x.tolist(-999)
        [[1, -999, 3], [-999, 5, -999], [7, -999, 9]]

        """
        _mask = self._mask
        if _mask is nomask:
            return self._data.tolist()
        if fill_value is not None:
            return self.filled(fill_value).tolist()
        names = self.dtype.names
        if names:
            result = self._data.astype([(_, object) for _ in names])
            for n in names:
                result[n][_mask[n]] = None
            return result.tolist()
        if _mask is nomask:
            return [None]
        inishape = self.shape
        result = np.array(self._data.ravel(), dtype=object)
        result[_mask.ravel()] = None
        result.shape = inishape
        return result.tolist()

    def tostring(self, fill_value=None, order='C'):
        """
        A compatibility alias for `tobytes`, with exactly the same behavior.

        Despite its name, it returns `bytes` not `str`\\ s.

        .. deprecated:: 1.19.0
        """
        warnings.warn('tostring() is deprecated. Use tobytes() instead.', DeprecationWarning, stacklevel=2)
        return self.tobytes(fill_value, order=order)

    def tobytes(self, fill_value=None, order='C'):
        """
        Return the array data as a string containing the raw bytes in the array.

        The array is filled with a fill value before the string conversion.

        .. versionadded:: 1.9.0

        Parameters
        ----------
        fill_value : scalar, optional
            Value used to fill in the masked values. Default is None, in which
            case `MaskedArray.fill_value` is used.
        order : {'C','F','A'}, optional
            Order of the data item in the copy. Default is 'C'.

            - 'C'   -- C order (row major).
            - 'F'   -- Fortran order (column major).
            - 'A'   -- Any, current order of array.
            - None  -- Same as 'A'.

        See Also
        --------
        numpy.ndarray.tobytes
        tolist, tofile

        Notes
        -----
        As for `ndarray.tobytes`, information about the shape, dtype, etc.,
        but also about `fill_value`, will be lost.

        Examples
        --------
        >>> x = np.ma.array(np.array([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
        >>> x.tobytes()
        b'\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00?B\\x0f\\x00\\x00\\x00\\x00\\x00?B\\x0f\\x00\\x00\\x00\\x00\\x00\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\x00'

        """
        return self.filled(fill_value).tobytes(order=order)

    def tofile(self, fid, sep='', format='%s'):
        """
        Save a masked array to a file in binary format.

        .. warning::
          This function is not implemented yet.

        Raises
        ------
        NotImplementedError
            When `tofile` is called.

        """
        raise NotImplementedError('MaskedArray.tofile() not implemented yet.')

    def toflex(self):
        """
        Transforms a masked array into a flexible-type array.

        The flexible type array that is returned will have two fields:

        * the ``_data`` field stores the ``_data`` part of the array.
        * the ``_mask`` field stores the ``_mask`` part of the array.

        Parameters
        ----------
        None

        Returns
        -------
        record : ndarray
            A new flexible-type `ndarray` with two fields: the first element
            containing a value, the second element containing the corresponding
            mask boolean. The returned record shape matches self.shape.

        Notes
        -----
        A side-effect of transforming a masked array into a flexible `ndarray` is
        that meta information (``fill_value``, ...) will be lost.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.toflex()
        array([[(1, False), (2,  True), (3, False)],
               [(4,  True), (5, False), (6,  True)],
               [(7, False), (8,  True), (9, False)]],
              dtype=[('_data', '<i8'), ('_mask', '?')])

        """
        ddtype = self.dtype
        _mask = self._mask
        if _mask is None:
            _mask = make_mask_none(self.shape, ddtype)
        mdtype = self._mask.dtype
        record = np.ndarray(shape=self.shape, dtype=[('_data', ddtype), ('_mask', mdtype)])
        record['_data'] = self._data
        record['_mask'] = self._mask
        return record
    torecords = toflex

    def __getstate__(self):
        """Return the internal state of the masked array, for pickling
        purposes.

        """
        cf = 'CF'[self.flags.fnc]
        data_state = super().__reduce__()[2]
        return data_state + (getmaskarray(self).tobytes(cf), self._fill_value)

    def __setstate__(self, state):
        """Restore the internal state of the masked array, for
        pickling purposes.  ``state`` is typically the output of the
        ``__getstate__`` output, and is a 5-tuple:

        - class name
        - a tuple giving the shape of the data
        - a typecode for the data
        - a binary string for the data
        - a binary string for the mask.

        """
        _, shp, typ, isf, raw, msk, flv = state
        super().__setstate__((shp, typ, isf, raw))
        self._mask.__setstate__((shp, make_mask_descr(typ), isf, msk))
        self.fill_value = flv

    def __reduce__(self):
        """Return a 3-tuple for pickling a MaskedArray.

        """
        return (_mareconstruct, (self.__class__, self._baseclass, (0,), 'b'), self.__getstate__())

    def __deepcopy__(self, memo=None):
        from copy import deepcopy
        copied = MaskedArray.__new__(type(self), self, copy=True)
        if memo is None:
            memo = {}
        memo[id(self)] = copied
        for k, v in self.__dict__.items():
            copied.__dict__[k] = deepcopy(v, memo)
        if self.dtype.hasobject:
            copied._data[...] = deepcopy(copied._data)
        return copied