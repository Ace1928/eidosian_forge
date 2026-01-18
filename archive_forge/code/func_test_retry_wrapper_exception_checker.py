from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
@mock.patch('oslo_db.api.time.sleep', return_value=None)
def test_retry_wrapper_exception_checker(self, mock_sleep):

    def exception_checker(exc):
        return isinstance(exc, ValueError) and exc.args[0] < 5

    @api.wrap_db_retry(max_retries=10, exception_checker=exception_checker)
    def some_method(res):
        res['result'] += 1
        raise ValueError(res['result'])
    res = {'result': 0}
    self.assertRaises(ValueError, some_method, res)
    self.assertEqual(5, res['result'])