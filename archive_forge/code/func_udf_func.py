import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def udf_func(ctx):

    class UDT:

        def __init__(self):
            self.caller = None

        def __call__(self, ctx):
            try:
                if self.caller is None:
                    self.caller, ctx = (batch_gen(ctx).send, None)
                batch = self.caller(ctx)
            except StopIteration:
                arrays = [pa.array([], type=field.type) for field in schema]
                batch = pa.RecordBatch.from_arrays(arrays=arrays, schema=schema)
            return batch.to_struct_array()
    return UDT()