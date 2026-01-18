import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
def test_nonetype_multiple_levels(self):
    data = {'id': None, 'location': {'id': None, 'country': {'id': None, 'state': {'id': None, 'town.info': {'region': None, 'x': 49.151580810546875, 'y': -33.148521423339844, 'z': 27.572303771972656}}}}}
    result = nested_to_record(data)
    expected = {'id': None, 'location.id': None, 'location.country.id': None, 'location.country.state.id': None, 'location.country.state.town.info.region': None, 'location.country.state.town.info.x': 49.151580810546875, 'location.country.state.town.info.y': -33.148521423339844, 'location.country.state.town.info.z': 27.572303771972656}
    assert result == expected