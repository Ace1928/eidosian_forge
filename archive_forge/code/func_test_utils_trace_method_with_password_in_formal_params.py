import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
def test_utils_trace_method_with_password_in_formal_params(self):
    mock_logging = self.mock_object(utils, 'logging')
    mock_log = mock.Mock()
    mock_log.isEnabledFor = lambda x: True
    mock_logging.getLogger = mock.Mock(return_value=mock_log)

    @utils.trace
    def _trace_test_method(*args, **kwargs):
        self.assertEqual('verybadpass', kwargs['connection']['data']['auth_password'])
        pass
    connector_properties = {'data': {'auth_password': 'verybadpass'}}
    _trace_test_method(self, connection=connector_properties)
    self.assertEqual(2, mock_log.debug.call_count)
    self.assertIn("'auth_password': '***'", str(mock_log.debug.call_args_list[0]))