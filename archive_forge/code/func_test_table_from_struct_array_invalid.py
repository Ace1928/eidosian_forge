from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_from_struct_array_invalid():
    with pytest.raises(TypeError, match="Argument 'struct_array' has incorrect type"):
        pa.Table.from_struct_array(pa.array(range(5)))