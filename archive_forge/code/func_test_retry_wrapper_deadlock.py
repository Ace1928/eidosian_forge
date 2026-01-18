from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
@mock.patch('oslo_db.api.time.sleep', return_value=None)
def test_retry_wrapper_deadlock(self, mock_sleep):

    @api.wrap_db_retry(max_retries=1, retry_on_deadlock=True)
    def some_method_no_deadlock():
        raise exception.RetryRequest(ValueError())
    with mock.patch('oslo_db.api.wrap_db_retry._get_inc_interval') as mock_get:
        mock_get.return_value = (2, 2)
        self.assertRaises(ValueError, some_method_no_deadlock)
        mock_get.assert_called_once_with(1, False)

    @api.wrap_db_retry(max_retries=1, retry_on_deadlock=True)
    def some_method_deadlock():
        raise exception.DBDeadlock('test')
    with mock.patch('oslo_db.api.wrap_db_retry._get_inc_interval') as mock_get:
        mock_get.return_value = (0.1, 2)
        self.assertRaises(exception.DBDeadlock, some_method_deadlock)
        mock_get.assert_called_once_with(1, True)

    @api.wrap_db_retry(max_retries=1, retry_on_deadlock=True, jitter=True)
    def some_method_no_deadlock_exp():
        raise exception.RetryRequest(ValueError())
    with mock.patch('oslo_db.api.wrap_db_retry._get_inc_interval') as mock_get:
        mock_get.return_value = (0.1, 2)
        self.assertRaises(ValueError, some_method_no_deadlock_exp)
        mock_get.assert_called_once_with(1, True)