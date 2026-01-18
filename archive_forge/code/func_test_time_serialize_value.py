import pytest
import datetime
import pytz
from traitlets import TraitError
from ..trait_types import (
def test_time_serialize_value():
    t = datetime.time(13, 37, 42, 7000)
    assert time_to_json(t, None) == dict(hours=13, minutes=37, seconds=42, milliseconds=7)