from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
class DBReconnectTestCase(DBAPITestCase):

    def setUp(self):
        super().setUp()
        self.test_db_api = DBAPI()
        patcher = mock.patch(__name__ + '.get_backend', return_value=self.test_db_api)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_raise_connection_error(self):
        self.dbapi = api.DBAPI('sqlalchemy', {'sqlalchemy': __name__})
        self.test_db_api.error_counter = 5
        self.assertRaises(exception.DBConnectionError, self.dbapi._api_raise)

    def test_raise_connection_error_decorated(self):
        self.dbapi = api.DBAPI('sqlalchemy', {'sqlalchemy': __name__})
        self.test_db_api.error_counter = 5
        self.assertRaises(exception.DBConnectionError, self.dbapi.api_raise_enable_retry)
        self.assertEqual(4, self.test_db_api.error_counter, 'Unexpected retry')

    def test_raise_connection_error_enabled(self):
        self.dbapi = api.DBAPI('sqlalchemy', {'sqlalchemy': __name__}, use_db_reconnect=True)
        self.test_db_api.error_counter = 5
        self.assertRaises(exception.DBConnectionError, self.dbapi.api_raise_default)
        self.assertEqual(4, self.test_db_api.error_counter, 'Unexpected retry')

    @mock.patch('oslo_db.api.time.sleep', return_value=None)
    def test_retry_one(self, p_time_sleep):
        self.dbapi = api.DBAPI('sqlalchemy', {'sqlalchemy': __name__}, use_db_reconnect=True, retry_interval=1)
        try:
            func = self.dbapi.api_raise_enable_retry
            self.test_db_api.error_counter = 1
            self.assertTrue(func(), 'Single retry did not succeed.')
        except Exception:
            self.fail('Single retry raised an un-wrapped error.')
        p_time_sleep.assert_called_with(1)
        self.assertEqual(0, self.test_db_api.error_counter, 'Counter not decremented, retry logic probably failed.')

    @mock.patch('oslo_db.api.time.sleep', return_value=None)
    def test_retry_two(self, p_time_sleep):
        self.dbapi = api.DBAPI('sqlalchemy', {'sqlalchemy': __name__}, use_db_reconnect=True, retry_interval=1, inc_retry_interval=False)
        try:
            func = self.dbapi.api_raise_enable_retry
            self.test_db_api.error_counter = 2
            self.assertTrue(func(), 'Multiple retry did not succeed.')
        except Exception:
            self.fail('Multiple retry raised an un-wrapped error.')
        p_time_sleep.assert_called_with(1)
        self.assertEqual(0, self.test_db_api.error_counter, 'Counter not decremented, retry logic probably failed.')

    @mock.patch('oslo_db.api.time.sleep', return_value=None)
    def test_retry_float_interval(self, p_time_sleep):
        self.dbapi = api.DBAPI('sqlalchemy', {'sqlalchemy': __name__}, use_db_reconnect=True, retry_interval=0.5)
        try:
            func = self.dbapi.api_raise_enable_retry
            self.test_db_api.error_counter = 1
            self.assertTrue(func(), 'Single retry did not succeed.')
        except Exception:
            self.fail('Single retry raised an un-wrapped error.')
        p_time_sleep.assert_called_with(0.5)
        self.assertEqual(0, self.test_db_api.error_counter, 'Counter not decremented, retry logic probably failed.')

    @mock.patch('oslo_db.api.time.sleep', return_value=None)
    def test_retry_until_failure(self, p_time_sleep):
        self.dbapi = api.DBAPI('sqlalchemy', {'sqlalchemy': __name__}, use_db_reconnect=True, retry_interval=1, inc_retry_interval=False, max_retries=3)
        func = self.dbapi.api_raise_enable_retry
        self.test_db_api.error_counter = 5
        self.assertRaises(exception.DBError, func, 'Retry of permanent failure did not throw DBError exception.')
        p_time_sleep.assert_called_with(1)
        self.assertNotEqual(0, self.test_db_api.error_counter, 'Retry did not stop after sql_max_retries iterations.')