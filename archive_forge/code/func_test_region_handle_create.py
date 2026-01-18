from unittest import mock
from urllib import parse
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import region
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_region_handle_create(self):
    mock_region = self._get_mock_region()
    self.regions.create.return_value = mock_region
    self.assertEqual('test_region_1', self.test_region.properties.get(region.KeystoneRegion.ID))
    self.assertEqual('Test region', self.test_region.properties.get(region.KeystoneRegion.DESCRIPTION))
    self.assertEqual('default_region', self.test_region.properties.get(region.KeystoneRegion.PARENT_REGION))
    self.assertEqual(True, self.test_region.properties.get(region.KeystoneRegion.ENABLED))
    self.test_region.handle_create()
    self.regions.create.assert_called_once_with(id=parse.quote('test_region_1'), description='Test region', parent_region='default_region', enabled=True)
    self.assertEqual(mock_region.id, self.test_region.resource_id)