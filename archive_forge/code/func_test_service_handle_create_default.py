import copy
from unittest import mock
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.keystone import service
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_service_handle_create_default(self):
    rsrc = self._setup_service_resource('test_create_with_defaults', use_default=True)
    mock_service = self._get_mock_service()
    self.services.create.return_value = mock_service
    rsrc.physical_resource_name = mock.MagicMock()
    rsrc.physical_resource_name.return_value = 'foo'
    self.assertIsNone(rsrc.properties.get(service.KeystoneService.NAME))
    self.assertIsNone(rsrc.properties.get(service.KeystoneService.DESCRIPTION))
    self.assertEqual('orchestration', rsrc.properties.get(service.KeystoneService.TYPE))
    self.assertTrue(rsrc.properties.get(service.KeystoneService.ENABLED))
    rsrc.handle_create()
    self.services.create.assert_called_once_with(name='foo', description=None, type='orchestration', enabled=True)