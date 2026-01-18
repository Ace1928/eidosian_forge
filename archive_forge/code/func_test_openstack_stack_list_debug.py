import yaml
from tempest.lib import exceptions
from heatclient.tests.functional.osc.v1 import base
def test_openstack_stack_list_debug(self):
    self.openstack('stack list', flags='--debug')