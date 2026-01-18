from novaclient import api_versions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import quotas as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def test_update_quota_invalid_resources(self):
    """Tests trying to update quota values for invalid resources."""
    q = self.cs.quotas.get('test')
    self.assertRaises(TypeError, q.update, floating_ips=1)
    self.assertRaises(TypeError, q.update, fixed_ips=1)
    self.assertRaises(TypeError, q.update, security_groups=1)
    self.assertRaises(TypeError, q.update, security_group_rules=1)
    self.assertRaises(TypeError, q.update, networks=1)
    self.assertRaises(TypeError, q.update, injected_files=1)
    self.assertRaises(TypeError, q.update, injected_file_content_bytes=1)
    self.assertRaises(TypeError, q.update, injected_file_path_bytes=1)