from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_show_no_fields(self):
    command = 'baremetal node show {} --fields {}'.format(self.node['uuid'], self.api_version)
    ex_text = 'expected at least one argument'
    self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)