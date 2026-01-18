from unittest import mock
import urllib
from glance.common import exception
from glance.common.scripts import utils as script_utils
import glance.tests.utils as test_utils
def test_validate_location_unsupported_error(self):
    location = 'swift'
    self.assertRaises(urllib.error.URLError, script_utils.validate_location_uri, location)
    location = 'swift+http'
    self.assertRaises(urllib.error.URLError, script_utils.validate_location_uri, location)
    location = 'swift+https'
    self.assertRaises(urllib.error.URLError, script_utils.validate_location_uri, location)
    location = 'swift+config'
    self.assertRaises(urllib.error.URLError, script_utils.validate_location_uri, location)
    location = 'vsphere'
    self.assertRaises(urllib.error.URLError, script_utils.validate_location_uri, location)
    location = 'rbd://'
    self.assertRaises(urllib.error.URLError, script_utils.validate_location_uri, location)
    location = 'cinder://'
    self.assertRaises(urllib.error.URLError, script_utils.validate_location_uri, location)