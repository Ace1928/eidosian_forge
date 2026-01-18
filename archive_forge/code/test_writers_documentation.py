from datetime import (
from functools import partial
from io import BytesIO
import os
import re
import numpy as np
import pytest
from pandas.compat import is_platform_windows
from pandas.compat._constants import PY310
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.excel import (
from pandas.io.excel._util import _writers

    Fixture to set engine for use in each test case.

    Rather than requiring `engine=...` to be provided explicitly as an
    argument in each test, this fixture sets a global option to dictate
    which engine should be used to write Excel files. After executing
    the test it rolls back said change to the global option.
    