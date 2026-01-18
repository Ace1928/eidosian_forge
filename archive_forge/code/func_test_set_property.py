import ddt
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
@ddt.data(('--property', '', 'error: the following arguments are required: <node>'), ('--property', 'prop', 'Attributes must be a list of PATH=VALUE'))
@ddt.unpack
def test_set_property(self, argument, value, ex_text):
    """Negative test for baremetal node set command options."""
    base_cmd = 'baremetal node set'
    command = self.construct_cmd(base_cmd, argument, value, self.node['uuid'])
    self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)