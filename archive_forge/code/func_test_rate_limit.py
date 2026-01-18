import io
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_log import rate_limit
@mock.patch('oslo_log.rate_limit.monotonic_clock')
def test_rate_limit(self, mock_clock):
    mock_clock.return_value = 1
    logger, stream = self.install_filter(2, 1)
    logger.error('message 1')
    logger.error('message 2')
    logger.error('message 3')
    self.assertEqual(stream.getvalue(), 'message 1\nmessage 2\nLogging rate limit: drop after 2 records/1 sec\n')
    stream.seek(0)
    stream.truncate()
    mock_clock.return_value = 2
    logger.error('message 4')
    logger.error('message 5')
    logger.error('message 6')
    self.assertEqual(stream.getvalue(), 'message 4\nmessage 5\nLogging rate limit: drop after 2 records/1 sec\n')