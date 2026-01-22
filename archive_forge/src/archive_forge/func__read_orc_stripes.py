from __future__ import annotations
import pyarrow as pa
import pyarrow.orc as orc
from dask.dataframe.io.utils import _get_pyarrow_dtypes, _meta_from_dtypes
def _read_orc_stripes(fs, path, stripes, schema, columns):
    if columns is None:
        columns = list(schema)
    batches = []
    with fs.open(path, 'rb') as f:
        o = orc.ORCFile(f)
        _stripes = range(o.nstripes) if stripes is None else stripes
        for stripe in _stripes:
            batches.append(o.read_stripe(stripe, columns))
    return batches