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
def test_plugins_are_all_opts(self):
    manager = stevedore.ExtensionManager(loading.PLUGIN_NAMESPACE, propagate_map_exceptions=True)

    def inner(driver):
        for p in driver.plugin().get_options():
            self.assertIsInstance(p, loading.Opt)
    manager.map(inner)