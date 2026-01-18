from __future__ import annotations
from io import (
import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
def test_unsuported_compression(parser, geom_df):
    with pytest.raises(ValueError, match='Unrecognized compression type'):
        with tm.ensure_clean() as path:
            geom_df.to_xml(path, parser=parser, compression='7z')