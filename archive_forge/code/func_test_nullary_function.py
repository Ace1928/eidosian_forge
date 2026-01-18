import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_nullary_function(nullary_func_fixture):
    check_scalar_function(nullary_func_fixture, [], run_in_dataset=False, batch_length=1)