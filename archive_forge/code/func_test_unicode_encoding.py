from io import (
import os
import tempfile
import uuid
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_unicode_encoding(all_parsers, csv_dir_path):
    path = os.path.join(csv_dir_path, 'unicode_series.csv')
    parser = all_parsers
    result = parser.read_csv(path, header=None, encoding='latin-1')
    result = result.set_index(0)
    got = result[1][1632]
    expected = 'Á köldum klaka (Cold Fever) (1994)'
    assert got == expected