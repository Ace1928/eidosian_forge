import pytest
import datetime
import pytz
from traitlets import TraitError
from ..trait_types import (
def test_time_deserialize_none():
    assert time_from_json(None, None) == None