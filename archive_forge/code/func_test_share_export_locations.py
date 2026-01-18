from openstack.shared_file_system.v2 import share_export_locations as el
from openstack.tests.unit import base
def test_share_export_locations(self):
    export = el.ShareExportLocation(**EXAMPLE)
    self.assertEqual(EXAMPLE['id'], export.id)
    self.assertEqual(EXAMPLE['path'], export.path)
    self.assertEqual(EXAMPLE['preferred'], export.is_preferred)
    self.assertEqual(EXAMPLE['share_instance_id'], export.share_instance_id)
    self.assertEqual(EXAMPLE['is_admin_only'], export.is_admin)