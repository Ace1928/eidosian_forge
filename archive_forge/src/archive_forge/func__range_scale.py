import numpy as np
from .casting import best_float, floor_exact, int_abs, shared_range, type_info
from .volumeutils import array_to_file, finite_range
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