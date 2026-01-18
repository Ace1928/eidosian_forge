from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_to_string_truncate(self):
    df = DataFrame([{'a': 'foo', 'b': 'bar', 'c': "let's make this a very VERY long line that is longer than the default 50 character limit", 'd': 1}, {'a': 'foo', 'b': 'bar', 'c': 'stuff', 'd': 1}])
    df.set_index(['a', 'b', 'c'])
    assert df.to_string() == "     a    b                                                                                         c  d\n0  foo  bar  let's make this a very VERY long line that is longer than the default 50 character limit  1\n1  foo  bar                                                                                     stuff  1"
    with option_context('max_colwidth', 20):
        assert df.to_string() == "     a    b                                                                                         c  d\n0  foo  bar  let's make this a very VERY long line that is longer than the default 50 character limit  1\n1  foo  bar                                                                                     stuff  1"
    assert df.to_string(max_colwidth=20) == "     a    b                    c  d\n0  foo  bar  let's make this ...  1\n1  foo  bar                stuff  1"