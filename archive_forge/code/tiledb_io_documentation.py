from __future__ import annotations
from dask.array import core
Save array to the TileDB storage format

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
    