from decimal import Decimal
from io import (
import mmap
import os
import tarfile
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.mark.slow
@pytest.mark.parametrize('encoding', [None, 'utf-8'])
def test_parse_trim_buffers(c_parser_only, encoding):
    parser = c_parser_only
    record_ = '9999-9,99:99,,,,ZZ,ZZ,,,ZZZ-ZZZZ,.Z-ZZZZ,-9.99,,,9.99,ZZZZZ,,-99,9,ZZZ-ZZZZ,ZZ-ZZZZ,,9.99,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,999,ZZZ-ZZZZ,,ZZ-ZZZZ,,,,,ZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZ,,,9,9,9,9,99,99,999,999,ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZ,9,ZZ-ZZZZ,9.99,ZZ-ZZZZ,ZZ-ZZZZ,,,,ZZZZ,,,ZZ,ZZ,,,,,,,,,,,,,9,,,999.99,999.99,,,ZZZZZ,,,Z9,,,,,,,ZZZ,ZZZ,,,,,,,,,,,ZZZZZ,ZZZZZ,ZZZ-ZZZZZZ,ZZZ-ZZZZZZ,ZZ-ZZZZ,ZZ-ZZZZ,ZZ-ZZZZ,ZZ-ZZZZ,,,999999,999999,ZZZ,ZZZ,,,ZZZ,ZZZ,999.99,999.99,,,,ZZZ-ZZZ,ZZZ-ZZZ,-9.99,-9.99,9,9,,99,,9.99,9.99,9,9,9.99,9.99,,,,9.99,9.99,,99,,99,9.99,9.99,,,ZZZ,ZZZ,,999.99,,999.99,ZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,,,ZZZZZ,ZZZZZ,ZZZ,ZZZ,9,9,,,,,,ZZZ-ZZZZ,ZZZ999Z,,,999.99,,999.99,ZZZ-ZZZZ,,,9.999,9.999,9.999,9.999,-9.999,-9.999,-9.999,-9.999,9.999,9.999,9.999,9.999,9.999,9.999,9.999,9.999,99999,ZZZ-ZZZZ,,9.99,ZZZ,,,,,,,,ZZZ,,,,,9,,,,9,,,,,,,,,,ZZZ-ZZZZ,ZZZ-ZZZZ,,ZZZZZ,ZZZZZ,ZZZZZ,ZZZZZ,,,9.99,,ZZ-ZZZZ,ZZ-ZZZZ,ZZ,999,,,,ZZ-ZZZZ,ZZZ,ZZZ,ZZZ-ZZZZ,ZZZ-ZZZZ,,,99.99,99.99,,,9.99,9.99,9.99,9.99,ZZZ-ZZZZ,,,ZZZ-ZZZZZ,,,,,-9.99,-9.99,-9.99,-9.99,,,,,,,,,ZZZ-ZZZZ,,9,9.99,9.99,99ZZ,,-9.99,-9.99,ZZZ-ZZZZ,,,,,,,ZZZ-ZZZZ,9.99,9.99,9999,,,,,,,,,,-9.9,Z/Z-ZZZZ,999.99,9.99,,999.99,ZZ-ZZZZ,ZZ-ZZZZ,9.99,9.99,9.99,9.99,9.99,9.99,,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ-ZZZZZ,ZZZ,ZZZ,ZZZ,ZZZ,9.99,,,-9.99,ZZ-ZZZZ,-999.99,,-9999,,999.99,,,,999.99,99.99,,,ZZ-ZZZZZZZZ,ZZ-ZZZZ-ZZZZZZZ,,,,ZZ-ZZ-ZZZZZZZZ,ZZZZZZZZ,ZZZ-ZZZZ,9999,999.99,ZZZ-ZZZZ,-9.99,-9.99,ZZZ-ZZZZ,99:99:99,,99,99,,9.99,,-99.99,,,,,,9.99,ZZZ-ZZZZ,-9.99,-9.99,9.99,9.99,,ZZZ,,,,,,,ZZZ,ZZZ,,,,,'
    chunksize, n_lines = (128, 2 * 128 + 15)
    csv_data = '\n'.join([record_] * n_lines) + '\n'
    row = tuple((val_ if val_ else np.nan for val_ in record_.split(',')))
    expected = DataFrame([row for _ in range(n_lines)], dtype=object, columns=None, index=None)
    with parser.read_csv(StringIO(csv_data), header=None, dtype=object, chunksize=chunksize, encoding=encoding) as chunks_:
        result = concat(chunks_, axis=0, ignore_index=True)
    tm.assert_frame_equal(result, expected)