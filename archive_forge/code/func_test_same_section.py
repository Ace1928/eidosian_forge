from unittest import mock
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
import stevedore
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.loading._plugins.identity import v2
from keystoneauth1.loading._plugins.identity import v3
from keystoneauth1.tests.unit.loading import utils
@utils.mock_plugin()
def test_same_section(self, m):
    opts = loading.get_auth_plugin_conf_options(utils.MockLoader())
    self.conf_fixture.register_opts(opts, group=self.GROUP)
    loading.register_auth_conf_options(self.conf_fixture.conf, group=self.GROUP)
    self.conf_fixture.config(auth_type=uuid.uuid4().hex, group=self.GROUP, **self.TEST_VALS)
    a = loading.load_auth_from_conf_options(self.conf_fixture.conf, self.GROUP)
    self.assertTestVals(a)