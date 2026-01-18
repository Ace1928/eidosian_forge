from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
def test_repr_truncates_terminal_size_full(self, monkeypatch):
    terminal_size = (80, 24)
    df = DataFrame(np.random.default_rng(2).random((1, 7)))
    monkeypatch.setattr('pandas.io.formats.format.get_terminal_size', lambda: terminal_size)
    assert '...' not in str(df)