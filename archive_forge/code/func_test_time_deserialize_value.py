import pytest
import datetime
import pytz
from traitlets import TraitError
from ..trait_types import (
def test_time_deserialize_value():
    v = dict(hours=13, minutes=37, seconds=42, milliseconds=7)
    assert time_from_json(v, None) == datetime.time(13, 37, 42, 7000)