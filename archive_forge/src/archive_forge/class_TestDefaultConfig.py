import os.path
import shutil
import fixtures
import oslo_middleware
from glance.api.middleware import context
from glance.common import config
from glance.tests import utils as test_utils
class TestDefaultConfig(test_utils.BaseTestCase):

    def setUp(self):
        super(TestDefaultConfig, self).setUp()
        self.CONF = config.cfg.CONF
        self.CONF.import_group('profiler', 'glance.common.wsgi')

    def test_osprofiler_disabled(self):
        self.assertFalse(self.CONF.profiler.enabled)
        self.assertFalse(self.CONF.profiler.trace_sqlalchemy)