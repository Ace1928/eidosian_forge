import ddt
from manilaclient.tests.functional import base
@ddt.data({'k': 'value'}, {'k' * 255: 'value'}, {'key': 'v'}, {'key': 'v' * 1023})
def test_set_metadata_min_max_sizes_of_keys_and_values(self, metadata):
    self.user_client.set_share_metadata(self.share['id'], metadata)
    get = self.user_client.get_share_metadata(self.share['id'])
    key = list(metadata.keys())[0]
    self.assertIn(key, get)
    self.assertEqual(metadata[key], get[key])