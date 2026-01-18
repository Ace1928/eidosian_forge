import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def valid_next_time(old_value, new_value, max_value, amount=1):
    current_value = int(old_value)
    next_value = int(new_value)
    if current_value == max_value - amount:
        assert next_value == 0
    else:
        assert next_value == current_value + amount
    return True