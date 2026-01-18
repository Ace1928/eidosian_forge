import yaml
from tempest.lib import exceptions
from heatclient.tests.functional.osc.v1 import base
def test_openstack_help_cmd(self):
    help_text = self.openstack('help stack list')
    lines = help_text.split('\n')
    self.assertFirstLineStartsWith(lines, 'usage: openstack stack list')