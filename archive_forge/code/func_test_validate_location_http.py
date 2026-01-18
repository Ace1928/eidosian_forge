from unittest import mock
import urllib
from glance.common import exception
from glance.common.scripts import utils as script_utils
import glance.tests.utils as test_utils
def test_validate_location_http(self):
    location = 'http://example.com'
    self.assertEqual(location, script_utils.validate_location_uri(location))