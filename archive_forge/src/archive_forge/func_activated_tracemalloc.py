from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc
import numpy as np
import pytest
from pandas._libs import hashtable as ht
import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin
@contextmanager
def activated_tracemalloc() -> Generator[None, None, None]:
    tracemalloc.start()
    try:
        yield
    finally:
        tracemalloc.stop()