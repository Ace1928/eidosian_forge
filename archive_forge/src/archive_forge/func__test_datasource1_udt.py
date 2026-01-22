import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def _test_datasource1_udt(func_maker):
    schema = datasource1_schema()
    func = func_maker()
    func_name = func_maker.__name__
    func_args = datasource1_args(func, func_name)
    pc.register_tabular_function(*func_args)
    n = 3
    for item in pc.call_tabular_function(func_name):
        n -= 1
        assert item == _record_batch_for_range(schema, n)