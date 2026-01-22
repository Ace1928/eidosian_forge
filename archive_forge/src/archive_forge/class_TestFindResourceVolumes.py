from unittest import mock
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit import utils as test_utils
from openstackclient.volume import client  # noqa
class TestFindResourceVolumes(test_utils.TestCase):

    def setUp(self):
        super(TestFindResourceVolumes, self).setUp()
        api = mock.Mock()
        api.client = mock.Mock()
        api.client.get = mock.Mock()
        resp = mock.Mock()
        body = {'volumes': [{'id': ID, 'display_name': NAME}]}
        api.client.get.side_effect = [Exception('Not found'), (resp, body)]
        self.manager = volumes.VolumeManager(api)

    def test_find(self):
        result = utils.find_resource(self.manager, NAME)
        self.assertEqual(ID, result.id)
        self.assertEqual(NAME, result.display_name)

    def test_not_find(self):
        self.assertRaises(exceptions.CommandError, utils.find_resource, self.manager, 'GeorgeMartin')