from cinderclient.tests.functional import base
def test_transfer_list(self):
    transfer_list = self.cinder('transfer-list')
    self.assertTableHeaders(transfer_list, ['ID', 'Volume ID', 'Name'])