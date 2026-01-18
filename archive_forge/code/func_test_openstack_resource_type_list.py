import yaml
from tempest.lib import exceptions
from heatclient.tests.functional.osc.v1 import base
def test_openstack_resource_type_list(self):
    ret = self.openstack('orchestration resource type list')
    rsrc_types = self.parser.listing(ret)
    self.assertTableStruct(rsrc_types, ['Resource Type'])