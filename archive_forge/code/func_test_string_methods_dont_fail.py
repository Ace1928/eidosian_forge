import re
import pytest
from pandas.core.indexes.frozen import FrozenList
def test_string_methods_dont_fail(self, container):
    repr(container)
    str(container)
    bytes(container)