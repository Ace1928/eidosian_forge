import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def varargs_function(ctx, first, *values):
    acc = first
    for val in values:
        acc = pc.call_function('add', [acc, val], memory_pool=ctx.memory_pool)
    return acc