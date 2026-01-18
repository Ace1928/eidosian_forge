from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
def test_usecols_single_string(all_parsers):
    parser = all_parsers
    data = 'foo, bar, baz\n1000, 2000, 3000\n4000, 5000, 6000'
    with pytest.raises(ValueError, match=_msg_validate_usecols_arg):
        parser.read_csv(StringIO(data), usecols='foo')