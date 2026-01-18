import ddt
from manilaclient.tests.functional import base
@ddt.data({'k': 'value'}, {'k' * 255: 'value'}, {'key': 'v'}, {'key': 'v' * 1023})
def test_update_metadata_min_max_sizes_of_keys_and_values(self, metadata):
    self.user_client.update_all_share_metadata(self.share['id'], metadata)
    get = self.user_client.get_share_metadata(self.share['id'])
    self.assertEqual(len(metadata), len(get))
    for key in metadata:
        self.assertIn(key, get)
        self.assertEqual(metadata[key], get[key])