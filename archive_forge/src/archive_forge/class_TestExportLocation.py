from openstack.tests.functional.shared_file_system import base
class TestExportLocation(base.BaseSharedFileSystemTest):
    min_microversion = '2.9'

    def setUp(self):
        super().setUp()
        self.SHARE_NAME = self.getUniqueString()
        my_share = self.create_share(name=self.SHARE_NAME, size=2, share_type='dhss_false', share_protocol='NFS', description=None)
        self.SHARE_ID = my_share.id

    def test_export_locations(self):
        exs = self.user_cloud.shared_file_system.export_locations(self.SHARE_ID)
        self.assertGreater(len(list(exs)), 0)
        for ex in exs:
            for attribute in ('id', 'path', 'share_instance_id', 'updated_at', 'created_at'):
                self.assertTrue(hasattr(ex, attribute))
                self.assertIsInstance(getattr(ex, attribute), 'str')
            for attribute in ('is_preferred', 'is_admin'):
                self.assertTrue(hasattr(ex, attribute))
                self.assertIsInstance(getattr(ex, attribute), 'bool')