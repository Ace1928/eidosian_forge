import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
def test_retries_once(self):
    self.counter = 0
    interval = 2
    backoff_rate = 2
    retries = 3
    with mock.patch.object(utils, '_time_sleep') as mock_sleep:

        @utils.retry(exception.VolumeDeviceNotFound, interval, retries, backoff_rate)
        def fails_once():
            self.counter += 1
            if self.counter < 2:
                raise exception.VolumeDeviceNotFound(device='fake')
            else:
                return 'success'
        ret = fails_once()
        self.assertEqual('success', ret)
        self.assertEqual(2, self.counter)
        self.assertEqual(1, mock_sleep.call_count)
        mock_sleep.assert_called_with(interval)