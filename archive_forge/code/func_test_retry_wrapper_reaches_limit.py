from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
@mock.patch('oslo_db.api.time.sleep', return_value=None)
def test_retry_wrapper_reaches_limit(self, mock_sleep):
    max_retries = 2

    @api.wrap_db_retry(max_retries=max_retries)
    def some_method(res):
        res['result'] += 1
        raise exception.RetryRequest(ValueError())
    res = {'result': 0}
    self.assertRaises(ValueError, some_method, res)
    self.assertEqual(max_retries + 1, res['result'])