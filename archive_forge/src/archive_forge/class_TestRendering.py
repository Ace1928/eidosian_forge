import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas._testing as tm
class TestRendering:

    def test_str(self, index):
        index.name = 'foo'
        assert "'foo'" in str(index)
        assert type(index).__name__ in str(index)