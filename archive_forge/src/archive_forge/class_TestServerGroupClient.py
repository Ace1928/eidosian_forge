from novaclient.tests.functional import base
class TestServerGroupClient(base.ClientTestBase):
    """Server groups v2.1 functional tests."""
    COMPUTE_API_VERSION = '2.1'

    def _create_sg(self, policy):
        sg_name = self.name_generate()
        output = self.nova('server-group-create %s %s' % (sg_name, policy))
        sg_id = self._get_column_value_from_single_row_table(output, 'Id')
        return sg_id

    def test_create_server_group(self):
        sg_id = self._create_sg('affinity')
        self.addCleanup(self.nova, 'server-group-delete %s' % sg_id)
        sg = self.nova('server-group-get %s' % sg_id)
        result = self._get_column_value_from_single_row_table(sg, 'Id')
        self.assertEqual(sg_id, result)

    def test_list_server_group(self):
        sg_id = self._create_sg('affinity')
        self.addCleanup(self.nova, 'server-group-delete %s' % sg_id)
        sg = self.nova('server-group-list')
        result = self._get_column_value_from_single_row_table(sg, 'Id')
        self.assertEqual(sg_id, result)

    def test_delete_server_group(self):
        sg_id = self._create_sg('affinity')
        sg = self.nova('server-group-get %s' % sg_id)
        result = self._get_column_value_from_single_row_table(sg, 'Id')
        self.assertIsNotNone(result)
        self.nova('server-group-delete %s' % sg_id)