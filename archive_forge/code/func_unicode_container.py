import re
import pytest
from pandas.core.indexes.frozen import FrozenList
@pytest.fixture
def unicode_container():
    return FrozenList(['א', 'ב', 'c'])