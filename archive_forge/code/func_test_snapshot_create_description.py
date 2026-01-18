from cinderclient.tests.functional import base
def test_snapshot_create_description(self):
    """Test steps:

        1) create volume in Setup()
        2) create snapshot with description
        3) check that snapshot has right description
        """
    description = 'test_description'
    snapshot = self.object_create('snapshot', params='--description {0} {1}'.format(description, self.volume['id']))
    self.assertEqual(description, snapshot['description'])
    self.object_delete('snapshot', snapshot['id'])
    self.check_object_deleted('snapshot', snapshot['id'])