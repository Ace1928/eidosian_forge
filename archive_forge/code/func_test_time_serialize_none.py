import pytest
import datetime
import pytz
from traitlets import TraitError
from ..trait_types import (
def test_time_serialize_none():
    assert time_to_json(None, None) == None