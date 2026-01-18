from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
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