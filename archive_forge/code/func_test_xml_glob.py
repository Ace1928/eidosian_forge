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
@pytest.mark.parametrize('filename', ['test_xml_glob.xml', 'test_xml_glob*.xml'])
def test_xml_glob(tmp_path, filename):
    data = test_data['int_data']
    df = pd.DataFrame(data)
    filename_param = filename
    with warns_that_defaulting_to_pandas() if filename_param == 'test_xml_glob.xml' else contextlib.nullcontext():
        df.modin.to_xml_glob(str(tmp_path / filename), index=False)
        read_df = pd.read_xml_glob(str(tmp_path / filename))
    read_df = read_df.reset_index(drop=True)
    df_equals(read_df, df)