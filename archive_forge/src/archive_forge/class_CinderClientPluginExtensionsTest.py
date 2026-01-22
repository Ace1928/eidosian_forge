from unittest import mock
import uuid
from cinderclient import exceptions as cinder_exc
from keystoneauth1 import exceptions as ks_exceptions
from heat.common import exception
from heat.engine.clients.os import cinder
from heat.tests import common
from heat.tests import utils
class CinderClientPluginExtensionsTest(CinderClientPluginTest):
    """Tests for extensions in cinderclient."""

    def test_has_no_extensions(self):
        self.cinder_client.list_extensions.show_all.return_value = []
        self.assertFalse(self.cinder_plugin.has_extension('encryption'))

    def test_has_no_interface_extensions(self):
        mock_extension = mock.Mock()
        p = mock.PropertyMock(return_value='os-xxxx')
        type(mock_extension).alias = p
        self.cinder_client.list_extensions.show_all.return_value = [mock_extension]
        self.assertFalse(self.cinder_plugin.has_extension('encryption'))

    def test_has_os_interface_extension(self):
        mock_extension = mock.Mock()
        p = mock.PropertyMock(return_value='encryption')
        type(mock_extension).alias = p
        self.cinder_client.list_extensions.show_all.return_value = [mock_extension]
        self.assertTrue(self.cinder_plugin.has_extension('encryption'))