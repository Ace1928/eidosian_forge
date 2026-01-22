Determination of spline kernel weights (adapted from SciPy)

See more verbose comments for each case there:
https://github.com/scipy/scipy/blob/eba29d69846ab1299976ff4af71c106188397ccc/scipy/ndimage/src/ni_splines.c#L7   # NOQA

``spline_weights_inline`` is a dict where the key is the spline order and the
value is the spline weight initialization code.
