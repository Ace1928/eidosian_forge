import json
import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_set_unset(self):
    """Check baremetal deploy template set and unset commands.

        Test steps:
        1) Create baremetal deploy template in setUp.
        2) Set extra data for deploy template.
        3) Check that baremetal deploy template extra data was set.
        4) Unset extra data for deploy template.
        5) Check that baremetal deploy template  extra data was unset.
        """
    extra_key = 'ext'
    extra_value = 'testdata'
    self.openstack('baremetal deploy template set --extra {0}={1} {2}'.format(extra_key, extra_value, self.template['uuid']))
    show_prop = self.deploy_template_show(self.template['uuid'], fields=['extra'])
    self.assertEqual(extra_value, show_prop['extra'][extra_key])
    self.openstack('baremetal deploy template unset --extra {0} {1}'.format(extra_key, self.template['uuid']))
    show_prop = self.deploy_template_show(self.template['uuid'], fields=['extra'])
    self.assertNotIn(extra_key, show_prop['extra'])