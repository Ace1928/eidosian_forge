import yaml
from tempest.lib import exceptions
from heatclient.tests.functional.osc.v1 import base
def test_openstack_stack_not_found(self):
    cmds = ['stack abandon', 'stack check', 'stack output list', 'stack resume', 'stack show', 'stack snapshot list', 'stack suspend', 'stack template show', 'stack cancel']
    for cmd in cmds:
        err = self.assertRaises(exceptions.CommandFailed, self.openstack, cmd + ' I-AM-NOT-FOUND')
        self.assertIn('Stack not found: I-AM-NOT-FOUND', str(err))