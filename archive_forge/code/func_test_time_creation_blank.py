import pytest
import datetime
import pytz
from traitlets import TraitError
from ..widget_datetime import NaiveDatetimePicker
def test_time_creation_blank():
    w = NaiveDatetimePicker()
    assert w.value is None