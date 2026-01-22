import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range
class SlopeArrayWriter(ArrayWriter):
    """ArrayWriter that can use scalefactor for writing arrays

    The scalefactor allows the array writer to write floats to int output
    types, and rescale larger ints to smaller.  It can therefore lose
    precision.

    It extends the ArrayWriter class with attribute:

    * slope

    and methods:

    * reset() - reset slope to default (not adapted to self.array)
    * calc_scale() - calculate slope to best write self.array
    """

    def __init__(self, array, out_dtype=None, calc_scale=True, scaler_dtype=np.float32, **kwargs):
        """Initialize array writer

        Parameters
        ----------
        array : array-like
            array-like object
        out_dtype : None or dtype
            dtype with which `array` will be written.  For this class,
            `out_dtype`` needs to be the same as the dtype of the input `array`
            or a swapped version of the same.
        calc_scale : {True, False}, optional
            Whether to calculate scaling for writing `array` on initialization.
            If False, then you can calculate this scaling with
            ``obj.calc_scale()`` - see examples
        scaler_dtype : dtype-like, optional
            specifier for numpy dtype for scaling
        \\*\\*kwargs : keyword arguments
            This class processes only:

            * nan2zero : bool, optional
              Whether to set NaN values to 0 when writing integer output.
              Defaults to True.  If False, NaNs get converted with numpy
              ``astype``, and the behavior is undefined.  Ignored for floating
              point output.

        Examples
        --------
        >>> arr = np.array([0, 254], np.uint8)
        >>> aw = SlopeArrayWriter(arr)
        >>> aw.slope
        1.0
        >>> aw = SlopeArrayWriter(arr, np.int8)
        >>> aw.slope
        2.0
        >>> aw = SlopeArrayWriter(arr, np.int8, calc_scale=False)
        >>> aw.slope
        1.0
        >>> aw.calc_scale()
        >>> aw.slope
        2.0
        """
        nan2zero = kwargs.pop('nan2zero', True)
        self._array = np.asanyarray(array)
        arr_dtype = self._array.dtype
        if out_dtype is None:
            out_dtype = arr_dtype
        else:
            out_dtype = np.dtype(out_dtype)
        self._out_dtype = out_dtype
        self.scaler_dtype = np.dtype(scaler_dtype)
        self.reset()
        self._nan2zero = nan2zero
        self._has_nan = None
        if calc_scale:
            self.calc_scale()

    def scaling_needed(self):
        """Checks if scaling is needed for input array

        Raises WriterError if no scaling possible.

        The rules are in the code, but:

        * If numpy will cast, return False (no scaling needed)
        * If input or output is an object or structured type, raise
        * If input is complex, raise
        * If the output is float, return False
        * If the input array is all zero, return False
        * If there is no finite value, return False (the writer will strip the
          non-finite values)
        * By now we are casting to (u)int. If the input type is a float, return
          True (we do need scaling)
        * Now input and output types are (u)ints. If the min and max in the
          data are within range of the output type, return False
        * Otherwise return True
        """
        if not super().scaling_needed():
            return False
        mn, mx = self.finite_range()
        return (mn, mx) != (np.inf, -np.inf)

    def reset(self):
        """Set object to values before any scaling calculation"""
        self.slope = 1.0
        self._finite_range = None
        self._scale_calced = False

    def _get_slope(self):
        return self._slope

    def _set_slope(self, val):
        self._slope = np.squeeze(self.scaler_dtype.type(val))
    slope = property(_get_slope, _set_slope, None, 'get/set slope')

    def calc_scale(self, force=False):
        """Calculate / set scaling for floats/(u)ints to (u)ints"""
        if not force and self._scale_calced:
            return
        self.reset()
        if not self.scaling_needed():
            return
        self._do_scaling()
        self._scale_calced = True

    def _writing_range(self):
        """Finite range for thresholding on write"""
        if self._out_dtype.kind in 'iu' and self._array.dtype.kind == 'f':
            mn, mx = self.finite_range()
            if (mn, mx) == (np.inf, -np.inf):
                mn, mx = (0, 0)
            return (mn, mx)
        return (None, None)

    def to_fileobj(self, fileobj, order='F'):
        """Write array into `fileobj`

        Parameters
        ----------
        fileobj : file-like object
        order : {'F', 'C'}
            order (Fortran or C) to which to write array
        """
        mn, mx = self._writing_range()
        array_to_file(self._array, fileobj, self._out_dtype, offset=None, divslope=self.slope, mn=mn, mx=mx, order=order, nan2zero=self._needs_nan2zero())

    def _do_scaling(self):
        arr = self._array
        out_dtype = self._out_dtype
        assert out_dtype.kind in 'iu'
        mn, mx = self.finite_range()
        if arr.dtype.kind == 'f':
            if self._nan2zero and self.has_nan:
                mn = min(mn, 0)
                mx = max(mx, 0)
            self._range_scale(mn, mx)
            return
        info = np.iinfo(out_dtype)
        out_max, out_min = (info.max, info.min)
        if int(mx) <= int(out_max) and int(mn) >= int(out_min):
            return
        self._iu2iu()

    def _iu2iu(self):
        mn, mx = self.finite_range()
        out_dt = self._out_dtype
        if out_dt.kind == 'u':
            o_min, o_max = shared_range(self.scaler_dtype, out_dt)
            if mx <= 0 and int_abs(mn) <= int(o_max):
                self.slope = -1.0
                return
        self._range_scale(mn, mx)

    def _range_scale(self, in_min, in_max):
        """Calculate scaling based on data range and output type"""
        out_dtype = self._out_dtype
        info = type_info(out_dtype)
        out_min, out_max = (info['min'], info['max'])
        big_float = best_float()
        if out_dtype.kind == 'f':
            out_min, out_max = np.array((out_min, out_max), dtype=big_float)
        else:
            out_min, out_max = (big_float(v) for v in (out_min, out_max))
        if self._out_dtype.kind == 'u':
            if in_min < 0 and in_max > 0:
                raise WriterError('Cannot scale negative and positive numbers to uint without intercept')
            if in_max <= 0:
                self.slope = in_min / out_max
            else:
                self.slope = in_max / out_max
            return
        mx_slope = in_max / out_max
        mn_slope = in_min / out_min
        self.slope = np.max([mx_slope, mn_slope])