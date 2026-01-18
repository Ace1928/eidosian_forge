import yaml
from tempest.lib import exceptions
from heatclient.tests.functional.osc.v1 import base
def test_openstack_empty_lists(self):
    cmds = ['software config', 'software deployment', 'stack']
    for cmd in cmds:
        self.openstack(cmd + ' list')