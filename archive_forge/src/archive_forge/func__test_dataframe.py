import io
import numpy as np
import pyarrow as pa
from pyarrow.tests import util
def _test_dataframe(size=10000, seed=0):
    import pandas as pd
    np.random.seed(seed)
    df = pd.DataFrame({'uint8': _random_integers(size, np.uint8), 'uint16': _random_integers(size, np.uint16), 'uint32': _random_integers(size, np.uint32), 'uint64': _random_integers(size, np.uint64), 'int8': _random_integers(size, np.int8), 'int16': _random_integers(size, np.int16), 'int32': _random_integers(size, np.int32), 'int64': _random_integers(size, np.int64), 'float32': np.random.randn(size).astype(np.float32), 'float64': np.arange(size, dtype=np.float64), 'bool': np.random.randn(size) > 0, 'strings': [util.rands(10) for i in range(size)], 'all_none': [None] * size, 'all_none_category': [None] * size})
    return df