from unittest import mock
from oslo_config import cfg
from osprofiler.drivers import jaeger
from osprofiler import opts
from osprofiler.tests import test
from jaeger_client import Config
def test_service_name_prefix(self):
    cfg.CONF.set_default('service_name_prefix', 'prx1', 'profiler_jaeger')
    self.assertEqual('prx1-pr1-svc1', self.driver._get_service_name(cfg.CONF, 'pr1', 'svc1'))