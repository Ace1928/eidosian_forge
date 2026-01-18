import ddt
from manilaclient.tests.functional import base
def test_set_and_replace_metadata(self):
    md = {'key8': 'value8'}
    share = self.create_share(metadata=md, cleanup_in_class=False, client=self.get_user_client())
    self.user_client.set_share_metadata(share['id'], {'key9': 'value9'})
    self.user_client.update_all_share_metadata(share['id'], {'key10': 'value10'})
    metadata = self.user_client.get_share_metadata(share['id'])
    self.assertEqual(1, len(metadata))
    self.assertIn('key10', metadata)
    self.assertEqual('value10', metadata['key10'])