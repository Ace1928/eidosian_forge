from collections.abc import Generator
import contextlib
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import accessor
Ensure that an attribute added to 'obj' during the test is
    removed when we're done
    