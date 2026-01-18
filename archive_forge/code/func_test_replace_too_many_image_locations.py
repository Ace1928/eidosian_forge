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
def test_replace_too_many_image_locations(self):
    self.config(image_location_quota=1)
    self.image.locations = [{'url': 'file:///fake.img.tar.gz', 'metadata': {}}]
    locations = [{'url': 'file:///fake1.img.tar.gz', 'metadata': {}}, {'url': 'file:///fake2.img.tar.gz', 'metadata': {}}, {'url': 'file:///fake3.img.tar.gz', 'metadata': {}}]
    exc = self.assertRaises(exception.ImageLocationLimitExceeded, setattr, self.image, 'locations', locations)
    self.assertIn('Attempted: 3, Maximum: 1', encodeutils.exception_to_unicode(exc))
    self.assertEqual(1, len(self.image.locations))