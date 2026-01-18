from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm

Tests that duplicate columns are handled appropriately when parsed by the
CSV engine. In general, the expected result is that they are either thoroughly
de-duplicated (if mangling requested) or ignored otherwise.
