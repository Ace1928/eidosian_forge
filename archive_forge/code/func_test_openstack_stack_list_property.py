import yaml
from tempest.lib import exceptions
from heatclient.tests.functional.osc.v1 import base
def test_openstack_stack_list_property(self):
    self.openstack('stack list --property id=123')