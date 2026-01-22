import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range
class SlopeInterArrayWriter(SlopeArrayWriter):
    """Array writer that can use slope and intercept to scale array

    The writer can subtract an intercept, and divided by a slope, in order to
    be able to convert floating point values into a (u)int range, or to convert
    larger (u)ints to smaller.

    It extends the ArrayWriter class with attributes:

    * inter
    * slope

    and methods:

    * reset() - reset inter, slope to default (not adapted to self.array)
    * calc_scale() - calculate inter, slope to best write self.array
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
            specifier for numpy dtype for slope, intercept
        \\*\\*kwargs : keyword arguments
            This class processes only:

            * nan2zero : bool, optional
              Whether to set NaN values to 0 when writing integer output.
              Defaults to True.  If False, NaNs get converted with numpy
              ``astype``, and the behavior is undefined.  Ignored for floating
              point output.

        Examples
        --------
        >>> arr = np.array([0, 255], np.uint8)
        >>> aw = SlopeInterArrayWriter(arr)
        >>> aw.slope, aw.inter
        (1.0, 0.0)
        >>> aw = SlopeInterArrayWriter(arr, np.int8)
        >>> (aw.slope, aw.inter) == (1.0, 128)
        True
        >>> aw = SlopeInterArrayWriter(arr, np.int8, calc_scale=False)
        >>> aw.slope, aw.inter
        (1.0, 0.0)
        >>> aw.calc_scale()
        >>> (aw.slope, aw.inter) == (1.0, 128)
        True
        """
        super().__init__(array, out_dtype, calc_scale, scaler_dtype, **kwargs)

    def reset(self):
        """Set object to values before any scaling calculation"""
        super().reset()
        self.inter = 0.0

    def _get_inter(self):
        return self._inter

    def _set_inter(self, val):
        self._inter = np.squeeze(self.scaler_dtype.type(val))
    inter = property(_get_inter, _set_inter, None, 'get/set inter')

    def to_fileobj(self, fileobj, order='F'):
        """Write array into `fileobj`

        Parameters
        ----------
        fileobj : file-like object
        order : {'F', 'C'}
            order (Fortran or C) to which to write array
        """
        mn, mx = self._writing_range()
        array_to_file(self._array, fileobj, self._out_dtype, offset=None, intercept=self.inter, divslope=self.slope, mn=mn, mx=mx, order=order, nan2zero=self._needs_nan2zero())

    def _iu2iu(self):
        mn, mx = (int(v) for v in self.finite_range())
        out_dtype = self._out_dtype
        o_min, o_max = (int(v) for v in shared_range(self.scaler_dtype, out_dtype))
        type_range = o_max - o_min
        mn2mx = mx - mn
        if mn2mx <= type_range:
            if o_min == 0:
                inter = floor_exact(mn - o_min, self.scaler_dtype)
            else:
                midpoint = mn + int(np.ceil(mn2mx / 2.0))
                inter = floor_exact(midpoint, self.scaler_dtype)
            int_inter = int(inter)
            assert mn - int_inter >= o_min
            if mx - int_inter <= o_max:
                self.inter = inter
                return
        super()._iu2iu()

    def _range_scale(self, in_min, in_max):
        """Calculate scaling, intercept based on data range and output type"""
        if in_max == in_min:
            self.slope = 1.0
            self.inter = in_min
            return
        big_float = best_float()
        in_dtype = self._array.dtype
        out_dtype = self._out_dtype
        working_dtype = self.scaler_dtype
        if in_dtype.kind == 'f':
            in_min, in_max = np.array([in_min, in_max], dtype=big_float)
            in_range = np.diff([in_min, in_max])
        else:
            in_min, in_max = (int(in_min), int(in_max))
            in_range = big_float(in_max - in_min)
            in_min, in_max = (big_float(v) for v in (in_min, in_max))
        if out_dtype.kind == 'f':
            info = type_info(out_dtype)
            out_min, out_max = (info['min'], info['max'])
        else:
            out_min, out_max = shared_range(working_dtype, out_dtype)
            out_min, out_max = np.array((out_min, out_max), dtype=big_float)
        assert [v.dtype.kind for v in (out_min, out_max)] == ['f', 'f']
        out_range = out_max - out_min
        '\n        Think of the input values as a line starting (left) at in_min and\n        ending (right) at in_max.\n\n        The output values will be a line starting at out_min and ending at\n        out_max.\n\n        We are going to match the input line to the output line by subtracting\n        `inter` then dividing by `slope`.\n\n        Slope must scale the input line to have the same length as the output\n        line.  We find this scale factor by dividing the input range (line\n        length) by the output range (line length)\n        '
        slope = in_range / out_range
        "\n        Now we know the slope, we need the intercept.  The intercept will be\n        such that:\n\n            (in_min - inter) / slope = out_min\n\n        Solving for the intercept:\n\n            inter = in_min - out_min * slope\n\n        We can also flip the sign of the slope.  In that case we match the\n        in_max to the out_min:\n\n            (in_max - inter_flipped) / -slope = out_min\n            inter_flipped = in_max + out_min * slope\n\n        When we reconstruct the data, we're going to do:\n\n            data = saved_data * slope + inter\n\n        We can't change the range of the saved data (the whole range of the\n        integer type) or the range of the output data (the values we input). We\n        can change the intermediate values ``saved_data * slope`` by choosing\n        the sign of the slope to match the in_min or in_max to the left or\n        right end of the saved data range.\n\n        If the out_dtype is signed int, then abs(out_min) = abs(out_max) + 1\n        and the absolute value and therefore precision for values at the left\n        and right of the saved data range are very similar (e.g. -128 * slope,\n        127 * slope respectively).\n\n        If the out_dtype is unsigned int, then the absolute value at the left\n        is 0 and the precision is much higher than for the right end of the\n        range (e.g. 0 * slope, 255 * slope).\n\n        If the out_dtype is unsigned int then we choose the sign of the slope\n        to match the smaller of the in_min, in_max to the zero end of the saved\n        range.\n        "
        if out_min == 0 and np.abs(in_max) < np.abs(in_min):
            inter = in_max + out_min * slope
            slope *= -1
        else:
            inter = in_min - out_min * slope
        self.inter = inter
        self.slope = slope
        if not np.all(np.isfinite([self.slope, self.inter])):
            raise ScalingError('Slope / inter not both finite')
        if not (0 in (in_min, in_max) and self._nan2zero and self.has_nan):
            return
        nan_fill_f = -self.inter / self.slope
        nan_fill_i = np.rint(nan_fill_f)
        if nan_fill_i == np.array(nan_fill_i, dtype=out_dtype):
            return
        self.inter = -np.clip(nan_fill_f, out_min, out_max) * self.slope
        nan_fill_i = np.rint(-self.inter / self.slope)
        assert nan_fill_i == np.array(nan_fill_i, dtype=out_dtype)