import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def wrong_output_datatype(ctx, array):
    return pc.call_function('add', [array, 1])