from unittest import mock
from oslo_log.fixture import logging_error as log_fixture
import testtools
import webob
import glance.api.common
from glance.common import exception
from glance.tests.unit import fixtures as glance_fixtures
class TestThreadPool(testtools.TestCase):

    def setUp(self):
        super().setUp()
        self.useFixture(glance_fixtures.WarningsFixture())
        self.useFixture(log_fixture.get_logging_handle_error_fixture())
        self.useFixture(glance_fixtures.StandardLogging())

    @mock.patch('glance.async_.get_threadpool_model')
    def test_get_thread_pool(self, mock_gtm):
        get_thread_pool = glance.api.common.get_thread_pool
        pool1 = get_thread_pool('pool1', size=123)
        get_thread_pool('pool2', size=456)
        pool1a = get_thread_pool('pool1')
        self.assertEqual(pool1, pool1a)
        mock_gtm.return_value.assert_has_calls([mock.call(123), mock.call(456)])

    @mock.patch('glance.async_.get_threadpool_model')
    def test_get_thread_pool_log(self, mock_gtm):
        with mock.patch.object(glance.api.common, 'LOG') as mock_log:
            glance.api.common.get_thread_pool('test-pool')
            mock_log.debug.assert_called_once_with('Initializing named threadpool %r', 'test-pool')