from cinderclient.tests.functional import base
def test_type_list(self):
    type_list = self.cinder('type-list')
    self.assertTableHeaders(type_list, ['ID', 'Name'])