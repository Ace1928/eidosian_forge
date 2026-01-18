import re
import pytest
from pandas.core.indexes.frozen import FrozenList
def test_tricky_container_to_bytes_raises(self, unicode_container):
    msg = "^'str' object cannot be interpreted as an integer$"
    with pytest.raises(TypeError, match=msg):
        bytes(unicode_container)