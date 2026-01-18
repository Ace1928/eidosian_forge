import ddt
from manilaclient.tests.functional import base
def test_set_and_add_metadata(self):
    md = {'key5': 'value5'}
    share = self.create_share(metadata=md, cleanup_in_class=False, client=self.get_user_client())
    self.user_client.set_share_metadata(share['id'], {'key6': 'value6'})
    self.user_client.set_share_metadata(share['id'], {'key7': 'value7'})
    metadata = self.user_client.get_share_metadata(share['id'])
    self.assertEqual(3, len(metadata))
    for i in (5, 6, 7):
        key = 'key%s' % i
        self.assertIn(key, metadata)
        self.assertEqual('value%s' % i, metadata[key])