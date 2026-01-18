from unittest import mock
from oslo_config import fixture
from osprofiler import opts
from osprofiler.tests import test
@mock.patch('osprofiler.web.enable')
@mock.patch('osprofiler.web.disable')
def test_web_trace_enabled(self, mock_disable, mock_enable):
    opts.set_defaults(self.conf_fixture.conf, enabled=True, hmac_keys='MY_KEY')
    opts.enable_web_trace(self.conf_fixture.conf)
    opts.disable_web_trace(self.conf_fixture.conf)
    mock_enable.assert_called_once_with('MY_KEY')
    mock_disable.assert_called_once_with()