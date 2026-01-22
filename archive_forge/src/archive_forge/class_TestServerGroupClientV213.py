from novaclient.tests.functional.v2.legacy import test_server_groups
class TestServerGroupClientV213(test_server_groups.TestServerGroupClient):
    """Server groups v2.13 functional tests."""
    COMPUTE_API_VERSION = '2.13'
    expected_metadata = True
    expected_policy_rules = False

    def test_create_server_group(self):
        sg_id = self._create_sg('affinity')
        self.addCleanup(self.nova, 'server-group-delete %s' % sg_id)
        sg = self.nova('server-group-get %s' % sg_id)
        result = self._get_column_value_from_single_row_table(sg, 'Id')
        self._get_column_value_from_single_row_table(sg, 'User Id')
        self._get_column_value_from_single_row_table(sg, 'Project Id')
        self.assertEqual(sg_id, result)
        self._get_column_value_from_single_row_table(sg, 'Metadata')
        self.assertIn('affinity', self._get_column_value_from_single_row_table(sg, 'Policies'))
        self.assertNotIn('Rules', sg)

    def test_list_server_groups(self):
        sg_id = self._create_sg('affinity')
        self.addCleanup(self.nova, 'server-group-delete %s' % sg_id)
        sg = self.nova('server-group-list')
        result = self._get_column_value_from_single_row_table(sg, 'Id')
        self._get_column_value_from_single_row_table(sg, 'User Id')
        self._get_column_value_from_single_row_table(sg, 'Project Id')
        self.assertEqual(sg_id, result)
        if self.expected_metadata:
            self._get_column_value_from_single_row_table(sg, 'Metadata')
        else:
            self.assertNotIn(sg, 'Metadata')
        if self.expected_policy_rules:
            self.assertEqual('affinity', self._get_column_value_from_single_row_table(sg, 'Policy'))
            self.assertEqual('{}', self._get_column_value_from_single_row_table(sg, 'Rules'))
        else:
            self.assertIn('affinity', self._get_column_value_from_single_row_table(sg, 'Policies'))
            self.assertNotIn('Rules', sg)

    def test_get_server_group(self):
        sg_id = self._create_sg('affinity')
        self.addCleanup(self.nova, 'server-group-delete %s' % sg_id)
        sg = self.nova('server-group-get %s' % sg_id)
        result = self._get_column_value_from_single_row_table(sg, 'Id')
        self._get_column_value_from_single_row_table(sg, 'User Id')
        self._get_column_value_from_single_row_table(sg, 'Project Id')
        self.assertEqual(sg_id, result)
        if self.expected_metadata:
            self._get_column_value_from_single_row_table(sg, 'Metadata')
        else:
            self.assertNotIn(sg, 'Metadata')
        if self.expected_policy_rules:
            self.assertEqual('affinity', self._get_column_value_from_single_row_table(sg, 'Policy'))
            self.assertEqual('{}', self._get_column_value_from_single_row_table(sg, 'Rules'))
        else:
            self.assertIn('affinity', self._get_column_value_from_single_row_table(sg, 'Policies'))
            self.assertNotIn('Rules', sg)