from unittest import mock
from urllib import parse
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import resource
from heat.engine.resources.openstack.keystone import region
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_region_get_live_state(self):
    self.test_region.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    mock_dict = mock.MagicMock()
    mock_dict.to_dict.return_value = {'parent_region_id': None, 'enabled': True, 'id': '79e4d02f8b454a7885c413d5d4297813', 'links': {'self': 'link'}, 'description': ''}
    self.regions.get.return_value = mock_dict
    reality = self.test_region.get_live_state(self.test_region.properties)
    expected = {'parent_region': None, 'enabled': True, 'description': ''}
    self.assertEqual(expected, reality)