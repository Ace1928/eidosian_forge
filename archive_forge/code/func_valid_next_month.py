import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def valid_next_month(old_month, new_month, old_year, new_year):
    assert valid_next_time(old_month, new_month, max_value=12, amount=1)
    if int(old_month) == 11:
        assert int(new_year) == int(old_year) + 1
    else:
        assert int(new_year) == int(old_year)
    return True