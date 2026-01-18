import re
import pytest
from pandas.core.indexes.frozen import FrozenList
def test_slicing_maintains_type(self, container, lst):
    result = container[1:2]
    expected = lst[1:2]
    self.check_result(result, expected)