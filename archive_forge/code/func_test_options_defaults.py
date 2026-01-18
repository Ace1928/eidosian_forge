from unittest import mock
from oslo_config import fixture
from osprofiler import opts
from osprofiler.tests import test
def test_options_defaults(self):
    opts.set_defaults(self.conf_fixture.conf)
    self.assertFalse(self.conf_fixture.conf.profiler.enabled)
    self.assertFalse(self.conf_fixture.conf.profiler.trace_sqlalchemy)
    self.assertEqual('SECRET_KEY', self.conf_fixture.conf.profiler.hmac_keys)
    self.assertFalse(opts.is_trace_enabled(self.conf_fixture.conf))
    self.assertFalse(opts.is_db_trace_enabled(self.conf_fixture.conf))