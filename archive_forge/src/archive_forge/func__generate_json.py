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
def _generate_json(file_name, nrows, ncols):
    data = np.random.rand(nrows, ncols)
    df = pandas.DataFrame(data, columns=[f'col{x}' for x in range(ncols)])
    df.to_json(file_name, lines=True, orient='records')