import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
class TestDatetime64Timezone(_DeprecationTestCase):
    """Parsing of datetime64 with timezones deprecated in 1.11.0, because
    datetime64 is now timezone naive rather than UTC only.

    It will be quite a while before we can remove this, because, at the very
    least, a lot of existing code uses the 'Z' modifier to avoid conversion
    from local time to UTC, even if otherwise it handles time in a timezone
    naive fashion.
    """

    def test_string(self):
        self.assert_deprecated(np.datetime64, args=('2000-01-01T00+01',))
        self.assert_deprecated(np.datetime64, args=('2000-01-01T00Z',))

    @pytest.mark.skipif(not _has_pytz, reason='The pytz module is not available.')
    def test_datetime(self):
        tz = pytz.timezone('US/Eastern')
        dt = datetime.datetime(2000, 1, 1, 0, 0, tzinfo=tz)
        self.assert_deprecated(np.datetime64, args=(dt,))