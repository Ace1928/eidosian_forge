from datetime import (
import operator
import numpy as np
import pytest
from pandas import Timestamp
import pandas._testing as tm
def test_rich_comparison_with_unsupported_type():

    class Inf:

        def __lt__(self, o):
            return False

        def __le__(self, o):
            return isinstance(o, Inf)

        def __gt__(self, o):
            return not isinstance(o, Inf)

        def __ge__(self, o):
            return True

        def __eq__(self, other) -> bool:
            return isinstance(other, Inf)
    inf = Inf()
    timestamp = Timestamp('2018-11-30')
    for left, right in [(inf, timestamp), (timestamp, inf)]:
        assert left > right or left < right
        assert left >= right or left <= right
        assert not left == right
        assert left != right