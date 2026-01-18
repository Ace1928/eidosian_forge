from io import (
import os
import platform
from urllib.error import URLError
import uuid
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_nonexistent_path(all_parsers):
    parser = all_parsers
    path = f'{uuid.uuid4()}.csv'
    msg = '\\[Errno 2\\]'
    with pytest.raises(FileNotFoundError, match=msg) as e:
        parser.read_csv(path)
    assert path == e.value.filename