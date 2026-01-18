import contextlib
import json
from pathlib import Path
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.experimental.pandas as pd
from modin.config import AsyncReadMode, Engine
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import try_cast_to_pandas
@pytest.mark.skipif(Engine.get() not in ('Ray', 'Unidist', 'Dask'), reason=f'{Engine.get()} does not have experimental API')
@pytest.mark.parametrize('set_async_read_mode', [False, True], indirect=True)
def test_read_evaluated_dict(set_async_read_mode):

    def _generate_evaluated_dict(file_name, nrows, ncols):
        result = {}
        keys = [f'col{x}' for x in range(ncols)]
        with open(file_name, mode='w') as _file:
            for i in range(nrows):
                data = np.random.rand(ncols)
                for idx, key in enumerate(keys):
                    result[key] = data[idx]
                _file.write(str(result))
                _file.write('\n')

    def _custom_parser(io_input, **kwargs):
        cat_list = []
        asin_list = []
        for line in io_input:
            obj = eval(line)
            cat_list.append(obj['col1'])
            asin_list.append(obj['col2'])
        return pandas.DataFrame({'col1': asin_list, 'col2': cat_list})

    def columns_callback(io_input, **kwargs):
        columns = None
        for line in io_input:
            columns = list(eval(line).keys())[1:3]
            break
        return columns
    with ensure_clean() as filename:
        _generate_evaluated_dict(filename, 64, 8)
        df1 = pd.read_custom_text(filename, columns=['col1', 'col2'], custom_parser=_custom_parser)
        assert df1.shape == (64, 2)
        df2 = pd.read_custom_text(filename, columns=columns_callback, custom_parser=_custom_parser)
        if AsyncReadMode.get():
            df_equals(df1, df2)
    if not AsyncReadMode.get():
        df_equals(df1, df2)