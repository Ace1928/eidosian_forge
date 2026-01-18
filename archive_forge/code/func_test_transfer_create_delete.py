from cinderclient.tests.functional import base
def test_transfer_create_delete(self):
    """Create and delete a volume transfer"""
    volume = self.object_create('volume', params='1')
    transfer = self.object_create('transfer', params=volume['id'])
    self.assert_object_details(self.TRANSFER_PROPERTY, transfer.keys())
    self.object_delete('transfer', transfer['id'])
    self.check_object_deleted('transfer', transfer['id'])
    self.object_delete('volume', volume['id'])
    self.check_object_deleted('volume', volume['id'])