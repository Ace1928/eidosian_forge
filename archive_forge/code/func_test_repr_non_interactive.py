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
def test_repr_non_interactive(self):
    df = DataFrame('hello', index=range(1000), columns=range(5))
    with option_context('mode.sim_interactive', False, 'display.width', 0, 'display.max_rows', 5000):
        assert not has_truncated_repr(df)
        assert not has_expanded_repr(df)