import pytest
from datashader.datashape import (
def test_option_in_parent():
    x = int64
    y = Option(float32)
    z = optionify(x, y, y)
    assert z == y