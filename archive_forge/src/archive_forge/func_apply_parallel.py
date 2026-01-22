import numpy
def apply_parallel(function, array, chunks=None, depth=0, mode=None, extra_arguments=(), extra_keywords=None, *, dtype=None, compute=None, channel_axis=None):
    """Map a function in parallel across an array.

    Split an array into possibly overlapping chunks of a given depth and
    boundary type, call the given function in parallel on the chunks, combine
    the chunks and return the resulting array.

    Parameters
    ----------
    function : function
        Function to be mapped which takes an array as an argument.
    array : numpy array or dask array
        Array which the function will be applied to.
    chunks : int, tuple, or tuple of tuples, optional
        A single integer is interpreted as the length of one side of a square
        chunk that should be tiled across the array.  One tuple of length
        ``array.ndim`` represents the shape of a chunk, and it is tiled across
        the array.  A list of tuples of length ``ndim``, where each sub-tuple
        is a sequence of chunk sizes along the corresponding dimension. If
        None, the array is broken up into chunks based on the number of
        available cpus. More information about chunks is in the documentation
        `here <https://dask.pydata.org/en/latest/array-design.html>`_. When
        `channel_axis` is not None, the tuples can be length ``ndim - 1`` and
        a single chunk will be used along the channel axis.
    depth : int or sequence of int, optional
        The depth of the added boundary cells. A tuple can be used to specify a
        different depth per array axis. Defaults to zero. When `channel_axis`
        is not None, and a tuple of length ``ndim - 1`` is provided, a depth of
        0 will be used along the channel axis.
    mode : {'reflect', 'symmetric', 'periodic', 'wrap', 'nearest', 'edge'}, optional
        Type of external boundary padding.
    extra_arguments : tuple, optional
        Tuple of arguments to be passed to the function.
    extra_keywords : dictionary, optional
        Dictionary of keyword arguments to be passed to the function.
    dtype : data-type or None, optional
        The data-type of the `function` output. If None, Dask will attempt to
        infer this by calling the function on data of shape ``(1,) * ndim``.
        For functions expecting RGB or multichannel data this may be
        problematic. In such cases, the user should manually specify this dtype
        argument instead.

        .. versionadded:: 0.18
           ``dtype`` was added in 0.18.
    compute : bool, optional
        If ``True``, compute eagerly returning a NumPy Array.
        If ``False``, compute lazily returning a Dask Array.
        If ``None`` (default), compute based on array type provided
        (eagerly for NumPy Arrays and lazily for Dask Arrays).
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

    Returns
    -------
    out : ndarray or dask Array
        Returns the result of the applying the operation.
        Type is dependent on the ``compute`` argument.

    Notes
    -----
    Numpy edge modes 'symmetric', 'wrap', and 'edge' are converted to the
    equivalent ``dask`` boundary modes 'reflect', 'periodic' and 'nearest',
    respectively.
    Setting ``compute=False`` can be useful for chaining later operations.
    For example region selection to preview a result or storing large data
    to disk instead of loading in memory.

    """
    try:
        import dask.array as da
    except ImportError:
        raise RuntimeError("Could not import 'dask'.  Please install using 'pip install dask'")
    if extra_keywords is None:
        extra_keywords = {}
    if compute is None:
        compute = not isinstance(array, da.Array)
    if channel_axis is not None:
        channel_axis = channel_axis % array.ndim
    if chunks is None:
        shape = array.shape
        try:
            from multiprocessing import cpu_count
            ncpu = cpu_count()
        except NotImplementedError:
            ncpu = 4
        if channel_axis is not None:
            spatial_shape = shape[:channel_axis] + shape[channel_axis + 1:]
            chunks = list(_get_chunks(spatial_shape, ncpu))
            chunks.insert(channel_axis, shape[channel_axis])
            chunks = tuple(chunks)
        else:
            chunks = _get_chunks(shape, ncpu)
    elif channel_axis is not None and len(chunks) == array.ndim - 1:
        chunks = list(chunks)
        chunks.insert(channel_axis, array.shape[channel_axis])
        chunks = tuple(chunks)
    if mode == 'wrap':
        mode = 'periodic'
    elif mode == 'symmetric':
        mode = 'reflect'
    elif mode == 'edge':
        mode = 'nearest'
    elif mode is None:
        mode = 'reflect'
    if channel_axis is not None:
        if numpy.isscalar(depth):
            depth = [depth] * (array.ndim - 1)
        depth = list(depth)
        if len(depth) == array.ndim - 1:
            depth.insert(channel_axis, 0)
        depth = tuple(depth)

    def wrapped_func(arr):
        return function(arr, *extra_arguments, **extra_keywords)
    darr = _ensure_dask_array(array, chunks=chunks)
    res = darr.map_overlap(wrapped_func, depth, boundary=mode, dtype=dtype)
    if compute:
        res = res.compute()
    return res