from collections.abc import Iterator
from functools import partial
from io import (
import os
from pathlib import Path
import re
import threading
from urllib.error import URLError
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import file_path_to_url
def test_wikipedia_states_table(self, datapath, flavor_read_html):
    data = datapath('io', 'data', 'html', 'wikipedia_states.html')
    assert os.path.isfile(data), f'{repr(data)} is not a file'
    assert os.path.getsize(data), f'{repr(data)} is an empty file'
    result = flavor_read_html(data, match='Arizona', header=1)[0]
    assert result.shape == (60, 12)
    assert 'Unnamed' in result.columns[-1]
    assert result['sq mi'].dtype == np.dtype('float64')
    assert np.allclose(result.loc[0, 'sq mi'], 665384.04)