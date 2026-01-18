import warnings
import mock
import pytest
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.packages.six.moves import xrange
from urllib3.response import HTTPResponse
from urllib3.util.retry import RequestHistory, Retry
@pytest.mark.freeze_time('2019-06-03 11:00:00', tz_offset=0)
@pytest.mark.parametrize('retry_after_header,respect_retry_after_header,sleep_duration', [('3600', True, 3600), ('3600', False, None), ('Mon, 3 Jun 2019 12:00:00 UTC', True, 3600), ('Mon, 3 Jun 2019 12:00:00 UTC', False, None), ('Mon, 3 Jun 2019 11:00:00 UTC', True, None), ('Mon, 3 Jun 2019 11:00:00 UTC', False, None), ('Mon, 03 Jun 2019 11:30:12 GMT', True, 1812), ('Monday, 03-Jun-19 11:30:12 GMT', True, 1812), ('Mon Jun  3 11:30:12 2019', True, 1812)])
@pytest.mark.parametrize('stub_timezone', ['UTC', 'Asia/Jerusalem', None], indirect=True)
@pytest.mark.usefixtures('stub_timezone')
def test_respect_retry_after_header_sleep(self, retry_after_header, respect_retry_after_header, sleep_duration):
    retry = Retry(respect_retry_after_header=respect_retry_after_header)
    with mock.patch('time.sleep') as sleep_mock:
        response = HTTPResponse(status=503, headers={'Retry-After': retry_after_header})
        retry.sleep(response)
        if respect_retry_after_header and sleep_duration is not None:
            sleep_mock.assert_called_with(sleep_duration)
        else:
            sleep_mock.assert_not_called()