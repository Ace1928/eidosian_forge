import shapely

    Dump a geometry to an open file.

    Parameters
    ----------
    ob :
        A geometry object of any type to be dumped to WKT.
    fp :
        A file-like object which implements a `write` method.
    trim : bool, default False
        Remove excess decimals from the WKT.
    rounding_precision : int
        Round output to the specified number of digits.
        Default behavior returns full precision.
    output_dimension : int, default 3
        Force removal of dimensions above the one specified.

    Returns
    -------
    None
    