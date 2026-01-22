from unittest import mock
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import images as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import images
class ImagesTest(utils.FixturedTestCase):
    client_fixture_class = client.V1
    data_fixture_class = data.V1

    @mock.patch('novaclient.base.Manager.alternate_service_type')
    def test_list_images(self, mock_alternate_service_type):
        il = self.cs.glance.list()
        self.assert_request_id(il, fakes.FAKE_REQUEST_ID_LIST)
        self.assert_called('GET', '/v2/images')
        for i in il:
            self.assertIsInstance(i, images.Image)
        self.assertEqual(2, len(il))
        mock_alternate_service_type.assert_called_once_with('image', allowed_types=('image',))