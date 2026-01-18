import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
@pytest.fixture
def weekdays_as_str():
    weekdays_str = '\n        Sun\n        Mon\n        Tue\n        Wed\n        Thu\n        Fri\n        Sat\n    '
    return weekdays_str