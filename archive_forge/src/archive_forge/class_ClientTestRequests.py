from requests_mock.contrib import fixture as rm_fixture
from glanceclient import client
from glanceclient.tests.unit.v2.fixtures import image_create_fixture
from glanceclient.tests.unit.v2.fixtures import image_list_fixture
from glanceclient.tests.unit.v2.fixtures import image_show_fixture
from glanceclient.tests.unit.v2.fixtures import schema_fixture
from glanceclient.tests import utils as testutils
from glanceclient.v2.image_schema import _BASE_SCHEMA
class ClientTestRequests(testutils.TestCase):
    """Client tests using the requests mock library."""

    def test_list_bad_image_schema(self):
        self.requests = self.useFixture(rm_fixture.Fixture())
        self.requests.get('http://example.com/v2/schemas/image', json=schema_fixture)
        self.requests.get(f'http://example.com/v2/images?limit={DEFAULT_PAGE_SIZE}', json=image_list_fixture)
        gc = client.Client(2.2, 'http://example.com/v2.1')
        images = gc.images.list()
        for image in images:
            pass

    def test_show_bad_image_schema(self):
        self.requests = self.useFixture(rm_fixture.Fixture())
        self.requests.get('http://example.com/v2/schemas/image', json=schema_fixture)
        self.requests.get('http://example.com/v2/images/%s' % image_show_fixture['id'], json=image_show_fixture)
        gc = client.Client(2.2, 'http://example.com/v2.1')
        img = gc.images.get(image_show_fixture['id'])
        self.assertEqual(image_show_fixture['checksum'], img['checksum'])

    def test_invalid_disk_format(self):
        self.requests = self.useFixture(rm_fixture.Fixture())
        self.requests.get('http://example.com/v2/schemas/image', json=_BASE_SCHEMA)
        self.requests.post('http://example.com/v2/images', json=image_create_fixture)
        self.requests.get('http://example.com/v2/images/%s' % image_show_fixture['id'], json=image_show_fixture)
        gc = client.Client(2.2, 'http://example.com/v2.1')
        fields = {'disk_format': 'qbull2'}
        try:
            gc.images.create(**fields)
            self.fail('Failed to raise exception when using bad disk format')
        except TypeError:
            pass

    def test_valid_disk_format(self):
        self.requests = self.useFixture(rm_fixture.Fixture())
        self.requests.get('http://example.com/v2/schemas/image', json=_BASE_SCHEMA)
        self.requests.post('http://example.com/v2/images', json=image_create_fixture)
        self.requests.get('http://example.com/v2/images/%s' % image_show_fixture['id'], json=image_show_fixture)
        gc = client.Client(2.2, 'http://example.com/v2.1')
        fields = {'disk_format': 'vhdx'}
        gc.images.create(**fields)