import ddt
from manilaclient.tests.functional import base
def test_set_metadata_in_share_creation(self):
    md = {'key1': 'value1', 'key2': 'value2'}
    share = self.create_share(metadata=md, client=self.get_user_client())
    metadata = self.user_client.get_share_metadata(share['id'])
    self.assertEqual(2, len(metadata))
    self.assertIn('key1', metadata)
    self.assertIn('key2', metadata)
    self.assertEqual(md['key1'], metadata['key1'])
    self.assertEqual(md['key2'], metadata['key2'])