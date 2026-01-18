import pytest
import srsly
from thinc.api import (
def test_simple_model_roundtrip_bytes():
    model = Maxout(5, 10, nP=2).initialize()
    b = model.get_param('b')
    b += 1
    data = model.to_bytes()
    b = model.get_param('b')
    b -= 1
    model = model.from_bytes(data)
    assert model.get_param('b')[0, 0] == 1