from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
def _setup_attribute_maps(self):
    self.extended_attributes = {'resource_one': {'one': 'first'}, 'resource_two': {'two': 'second'}}
    self.extension_attrs_map = {'resource_one': {'three': 'third'}}