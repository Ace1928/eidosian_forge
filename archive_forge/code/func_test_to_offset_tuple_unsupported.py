import re
import pytest
from pandas._libs.tslibs import (
def test_to_offset_tuple_unsupported():
    with pytest.raises(TypeError, match='pass as a string instead'):
        to_offset((5, 'T'))