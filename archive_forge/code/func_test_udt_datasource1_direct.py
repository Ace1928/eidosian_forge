import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_udt_datasource1_direct():
    _test_datasource1_udt(datasource1_direct)