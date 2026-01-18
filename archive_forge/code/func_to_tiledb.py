from __future__ import annotations
from dask.array import core
def to_tiledb(darray, uri, compute=True, return_stored=False, storage_options=None, key=None, **kwargs):
    """Save array to the TileDB storage format

    Save 'array' using the TileDB storage manager, to any TileDB-supported URI,
    including local disk, S3, or HDFS.

    See https://docs.tiledb.io for more information about TileDB.

    Parameters
    ----------

    darray: dask.array
        A dask array to write.
    uri:
        Any supported TileDB storage location.
    storage_options: dict
        Dict containing any configuration options for the TileDB backend.
        see https://docs.tiledb.io/en/stable/tutorials/config.html
    compute, return_stored: see ``store()``
    key: str or None
        Encryption key

    Returns
    -------

    None
        Unless ``return_stored`` is set to ``True`` (``False`` by default)

    Notes
    -----

    TileDB only supports regularly-chunked arrays.
    TileDB `tile extents`_ correspond to form 2 of the dask
    `chunk specification`_, and the conversion is
    done automatically for supported arrays.

    Examples
    --------

    >>> import dask.array as da, tempfile
    >>> uri = tempfile.NamedTemporaryFile().name
    >>> data = da.random.random(5,5)
    >>> da.to_tiledb(data, uri)
    >>> import tiledb
    >>> tdb_ar = tiledb.open(uri)
    >>> all(tdb_ar == data)
    True

    .. _chunk specification: https://docs.tiledb.io/en/stable/tutorials/tiling-dense.html
    .. _tile extents: http://docs.dask.org/en/latest/array-chunks.html
    """
    import tiledb
    tiledb_config = storage_options or dict()
    key = key or tiledb_config.pop('key', None)
    if not core._check_regular_chunks(darray.chunks):
        raise ValueError('Attempt to save array to TileDB with irregular chunking, please call `arr.rechunk(...)` first.')
    if isinstance(uri, str):
        chunks = [c[0] for c in darray.chunks]
        tdb = tiledb.empty_like(uri, darray, tile=chunks, config=tiledb_config, key=key, **kwargs)
    elif isinstance(uri, tiledb.Array):
        tdb = uri
        if not (darray.dtype == tdb.dtype and darray.ndim == tdb.ndim):
            raise ValueError('Target TileDB array layout is not compatible with source array')
    else:
        raise ValueError("'uri' must be string pointing to supported TileDB store location or an open, writable TileDB array.")
    if not (tdb.isopen and tdb.iswritable):
        raise ValueError('Target TileDB array is not open and writable.')
    return darray.store(tdb, lock=False, compute=compute, return_stored=return_stored)