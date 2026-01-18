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
def test_to_string_ascii_error(self):
    data = [('0  ', '                        .gitignore ', '     5 ', ' â\x80¢â\x80¢â\x80¢â\x80¢â\x80¢')]
    df = DataFrame(data)
    repr(df)