import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
class TestServersDetailsFlavorInfo(base.ClientTestBase):
    COMPUTE_API_VERSION = '2.47'

    def _validate_flavor_details(self, flavor_details, server_details):
        flavor_key_mapping = {'OS-FLV-EXT-DATA:ephemeral': 'flavor:ephemeral', 'disk': 'flavor:disk', 'extra_specs': 'flavor:extra_specs', 'name': 'flavor:original_name', 'ram': 'flavor:ram', 'swap': 'flavor:swap', 'vcpus': 'flavor:vcpus'}
        for key in flavor_key_mapping:
            flavor_val = self._get_value_from_the_table(flavor_details, key)
            server_flavor_val = self._get_value_from_the_table(server_details, flavor_key_mapping[key])
            if key == 'swap' and flavor_val == '':
                flavor_val = '0'
            self.assertEqual(flavor_val, server_flavor_val)

    def _setup_extra_specs(self, flavor_id):
        extra_spec_key = 'dummykey'
        self.nova('flavor-key', params='%(flavor)s set %(key)s=dummyval' % {'flavor': flavor_id, 'key': extra_spec_key})
        unset_params = '%(flavor)s unset %(key)s' % {'flavor': flavor_id, 'key': extra_spec_key}
        self.addCleanup(self.nova, 'flavor-key', params=unset_params)

    def test_show(self):
        self._setup_extra_specs(self.flavor.id)
        uuid = self._create_server().id
        server_output = self.nova('show %s' % uuid)
        flavor_output = self.nova('flavor-show %s' % self.flavor.id)
        self._validate_flavor_details(flavor_output, server_output)

    def test_show_minimal(self):
        uuid = self._create_server().id
        server_output = self.nova('show --minimal %s' % uuid)
        server_output_flavor = self._get_value_from_the_table(server_output, 'flavor')
        self.assertEqual(self.flavor.name, server_output_flavor)

    def test_list(self):
        self._setup_extra_specs(self.flavor.id)
        self._create_server()
        server_output = self.nova('list --fields flavor:disk')
        server_flavor_val = self._get_column_value_from_single_row_table(server_output, 'flavor: Disk')
        flavor_output = self.nova('flavor-show %s' % self.flavor.id)
        flavor_val = self._get_value_from_the_table(flavor_output, 'disk')
        self.assertEqual(flavor_val, server_flavor_val)