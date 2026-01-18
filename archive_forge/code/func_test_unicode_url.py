from .. import osutils, tests, urlutils
from ..directory_service import directories
from ..location import hooks as location_hooks
from ..location import location_to_url, rcp_location_to_url
def test_unicode_url(self):
    self.assertRaises(urlutils.InvalidURL, location_to_url, b'http://fo/\xc3\xaf'.decode('utf-8'))