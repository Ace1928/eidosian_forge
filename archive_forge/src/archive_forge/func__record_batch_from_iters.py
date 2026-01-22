import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def _record_batch_from_iters(schema, *iters):
    arrays = [pa.array(list(v), type=schema[i].type) for i, v in enumerate(iters)]
    return pa.RecordBatch.from_arrays(arrays=arrays, schema=schema)