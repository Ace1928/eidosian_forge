from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
def test_update_attributes_map_without_ext_attrs(self):
    base_attrs = {'ports': {'a': 'A'}}
    self.extn.update_attributes_map(base_attrs)
    self.assertIn('a', self.extn.get_extended_resources('2.0')['ports'])