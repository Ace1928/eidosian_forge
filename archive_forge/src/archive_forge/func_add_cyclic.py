import numpy as np
import numpy.ma as ma
def add_cyclic(data, x=None, y=None, axis=-1, cyclic=360, precision=0.0001):
    """
    Add a cyclic point to an array and optionally corresponding
    x/longitude and y/latitude coordinates.

    Checks all differences between the first and last
    x-coordinates along ``axis`` to be less than ``precision``.

    Parameters
    ----------
    data : ndarray
        An n-dimensional array of data to add a cyclic point to.
    x : ndarray, optional
        An n-dimensional array which specifies the x-coordinate values
        for the dimension the cyclic point is to be added to, i.e. normally the
        longitudes. Defaults to None.

        If ``x`` is given then *add_cyclic* checks if a cyclic point is
        already present by checking all differences between the first and last
        coordinates to be less than ``precision``.
        No point is added if a cyclic point was detected.

        If ``x`` is 1- or 2-dimensional, ``x.shape[-1]`` must equal
        ``data.shape[axis]``, otherwise ``x.shape[axis]`` must equal
        ``data.shape[axis]``.
    y : ndarray, optional
        An n-dimensional array with the values of the y-coordinate, i.e.
        normally the latitudes.
        The cyclic point simply copies the last column. Defaults to None.

        No cyclic point is added if ``y`` is 1-dimensional.
        If ``y`` is 2-dimensional, ``y.shape[-1]`` must equal
        ``data.shape[axis]``, otherwise ``y.shape[axis]`` must equal
        ``data.shape[axis]``.
    axis : int, optional
        Specifies the axis of the arrays to add the cyclic point to,
        i.e. axis with changing x-coordinates. Defaults to the right-most axis.
    cyclic : int or float, optional
        Width of periodic domain (default: 360).
    precision : float, optional
        Maximal difference between first and last x-coordinate to detect
        cyclic point (default: 1e-4).

    Returns
    -------
    cyclic_data
        The data array with a cyclic point added.
    cyclic_x
        The x-coordinate with a cyclic point, only returned if the ``x``
        keyword was supplied.
    cyclic_y
        The y-coordinate with the last column of the cyclic axis duplicated,
        only returned if ``x`` was 2- or n-dimensional and the ``y``
        keyword was supplied.

    Examples
    --------
    Adding a cyclic point to a data array, where the cyclic dimension is
    the right-most dimension.

    >>> import numpy as np
    >>> data = np.ones([5, 6]) * np.arange(6)
    >>> cyclic_data = add_cyclic(data)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]]

    Adding a cyclic point to a data array and an associated x-coordinate.

    >>> lons = np.arange(0, 360, 60)
    >>> cyclic_data, cyclic_lons = add_cyclic(data, x=lons)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]]
    >>> print(cyclic_lons)
    [  0  60 120 180 240 300 360]

    Adding a cyclic point to a data array and an associated 2-dimensional
    x-coordinate.

    >>> lons = np.arange(0, 360, 60)
    >>> lats = np.arange(-90, 90, 180/5)
    >>> lon2d, lat2d = np.meshgrid(lons, lats)
    >>> cyclic_data, cyclic_lon2d = add_cyclic(data, x=lon2d)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]]
    >>> print(cyclic_lon2d)
    [[  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]]

    Adding a cyclic point to a data array and the associated 2-dimensional
    x- and y-coordinates.

    >>> lons = np.arange(0, 360, 60)
    >>> lats = np.arange(-90, 90, 180/5)
    >>> lon2d, lat2d = np.meshgrid(lons, lats)
    >>> cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(
    ...     data, x=lon2d, y=lat2d)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]
     [0. 1. 2. 3. 4. 5. 0.]]
    >>> print(cyclic_lon2d)
    [[  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]
     [  0  60 120 180 240 300 360]]
    >>> print(cyclic_lat2d)
    [[-90. -90. -90. -90. -90. -90. -90.]
     [-54. -54. -54. -54. -54. -54. -54.]
     [-18. -18. -18. -18. -18. -18. -18.]
     [ 18.  18.  18.  18.  18.  18.  18.]
     [ 54.  54.  54.  54.  54.  54.  54.]]

    Not adding a cyclic point if cyclic point detected in x.

    >>> lons = np.arange(0, 361, 72)
    >>> lats = np.arange(-90, 90, 180/5)
    >>> lon2d, lat2d = np.meshgrid(lons, lats)
    >>> cyclic_data, cyclic_lon2d, cyclic_lat2d = add_cyclic(
    ...     data, x=lon2d, y=lat2d)
    >>> print(cyclic_data)  # doctest: +NORMALIZE_WHITESPACE
    [[0. 1. 2. 3. 4. 5.]
     [0. 1. 2. 3. 4. 5.]
     [0. 1. 2. 3. 4. 5.]
     [0. 1. 2. 3. 4. 5.]
     [0. 1. 2. 3. 4. 5.]]
    >>> print(cyclic_lon2d)
    [[  0  72 144 216 288 360]
     [  0  72 144 216 288 360]
     [  0  72 144 216 288 360]
     [  0  72 144 216 288 360]
     [  0  72 144 216 288 360]]
    >>> print(cyclic_lat2d)
    [[-90. -90. -90. -90. -90. -90.]
     [-54. -54. -54. -54. -54. -54.]
     [-18. -18. -18. -18. -18. -18.]
     [ 18.  18.  18.  18.  18.  18.]
     [ 54.  54.  54.  54.  54.  54.]]

    """
    if x is None:
        return _add_cyclic_data(data, axis=axis)
    if x.ndim > 2:
        xaxis = axis
    else:
        xaxis = -1
    if x.shape[xaxis] != data.shape[axis]:
        estr = f'x.shape[{xaxis}] does not match the size of the corresponding dimension of the data array: x.shape[{xaxis}] = {x.shape[xaxis]}, data.shape[{axis}] = {data.shape[axis]}.'
        raise ValueError(estr)
    if has_cyclic(x, axis=xaxis, cyclic=cyclic, precision=precision):
        if y is None:
            return (data, x)
        return (data, x, y)
    out_data = _add_cyclic_data(data, axis=axis)
    out_x = _add_cyclic_x(x, axis=xaxis, cyclic=cyclic)
    if y is None:
        return (out_data, out_x)
    if y.ndim == 1:
        return (out_data, out_x, y)
    if y.ndim > 2:
        yaxis = axis
    else:
        yaxis = -1
    if y.shape[yaxis] != data.shape[axis]:
        estr = f'y.shape[{yaxis}] does not match the size of the corresponding dimension of the data array: y.shape[{yaxis}] = {y.shape[yaxis]}, data.shape[{axis}] = {data.shape[axis]}.'
        raise ValueError(estr)
    out_y = _add_cyclic_data(y, axis=yaxis)
    return (out_data, out_x, out_y)