from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
@mock.patch('oslo_db.api.LOG')
def test_retry_wrapper_non_db_error_not_logged(self, mock_log):

    @api.wrap_db_retry(max_retries=5, retry_on_deadlock=True)
    def some_method():
        raise AttributeError('test')
    self.assertRaises(AttributeError, some_method)
    self.assertFalse(mock_log.called)