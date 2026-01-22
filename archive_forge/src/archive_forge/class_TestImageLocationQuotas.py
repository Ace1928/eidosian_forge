import copy
import fixtures
from unittest import mock
from unittest.mock import patch
import uuid
from oslo_limit import exception as ol_exc
from oslo_utils import encodeutils
from oslo_utils import units
from glance.common import exception
from glance.common import store_utils
import glance.quota
from glance.quota import keystone as ks_quota
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
class TestImageLocationQuotas(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImageLocationQuotas, self).setUp()
        self.base_image = mock.Mock()
        self.base_image.locations = []
        self.base_image.size = 1
        self.base_image.extra_properties = {}
        self.image = glance.quota.ImageProxy(self.base_image, mock.Mock(), mock.Mock(), mock.Mock())
        self.image_repo_mock = mock.Mock()
        self.image_repo_proxy = glance.quota.ImageRepoProxy(self.image_repo_mock, mock.Mock(), mock.Mock(), mock.Mock())

    def test_replace_image_location(self):
        self.config(image_location_quota=1)
        self.image.locations = [{'url': 'file:///fake.img.tar.gz', 'metadata': {}}]
        self.assertEqual(1, len(self.image.locations))

    def test_replace_too_many_image_locations(self):
        self.config(image_location_quota=1)
        self.image.locations = [{'url': 'file:///fake.img.tar.gz', 'metadata': {}}]
        locations = [{'url': 'file:///fake1.img.tar.gz', 'metadata': {}}, {'url': 'file:///fake2.img.tar.gz', 'metadata': {}}, {'url': 'file:///fake3.img.tar.gz', 'metadata': {}}]
        exc = self.assertRaises(exception.ImageLocationLimitExceeded, setattr, self.image, 'locations', locations)
        self.assertIn('Attempted: 3, Maximum: 1', encodeutils.exception_to_unicode(exc))
        self.assertEqual(1, len(self.image.locations))

    def test_replace_unlimited_image_locations(self):
        self.config(image_location_quota=-1)
        self.image.locations = [{'url': 'file:///fake.img.tar.gz', 'metadata': {}}]
        self.assertEqual(1, len(self.image.locations))

    def test_add_image_location(self):
        self.config(image_location_quota=1)
        location = {'url': 'file:///fake.img.tar.gz', 'metadata': {}}
        self.image.locations.append(location)
        self.assertEqual(1, len(self.image.locations))

    def test_add_too_many_image_locations(self):
        self.config(image_location_quota=1)
        location1 = {'url': 'file:///fake1.img.tar.gz', 'metadata': {}}
        self.image.locations.append(location1)
        location2 = {'url': 'file:///fake2.img.tar.gz', 'metadata': {}}
        exc = self.assertRaises(exception.ImageLocationLimitExceeded, self.image.locations.append, location2)
        self.assertIn('Attempted: 2, Maximum: 1', encodeutils.exception_to_unicode(exc))

    def test_add_unlimited_image_locations(self):
        self.config(image_location_quota=-1)
        location1 = {'url': 'file:///fake1.img.tar.gz', 'metadata': {}}
        self.image.locations.append(location1)
        self.assertEqual(1, len(self.image.locations))

    def test_remove_image_location_while_over_quota(self):
        self.config(image_location_quota=1)
        location1 = {'url': 'file:///fake1.img.tar.gz', 'metadata': {}}
        self.image.locations.append(location1)
        self.assertEqual(1, len(self.image.locations))
        self.config(image_location_quota=0)
        self.image.locations.remove(location1)
        self.assertEqual(0, len(self.image.locations))