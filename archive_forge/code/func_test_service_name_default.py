from unittest import mock
from oslo_config import cfg
from osprofiler.drivers import jaeger
from osprofiler import opts
from osprofiler.tests import test
from jaeger_client import Config
def test_service_name_default(self):
    self.assertEqual('pr1-svc1', self.driver._get_service_name(cfg.CONF, 'pr1', 'svc1'))