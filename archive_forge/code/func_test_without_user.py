from .. import osutils, tests, urlutils
from ..directory_service import directories
from ..location import hooks as location_hooks
from ..location import location_to_url, rcp_location_to_url
def test_without_user(self):
    self.assertEqual('git+ssh://example.com/srv/git/bar', rcp_location_to_url('example.com:/srv/git/bar', scheme='git+ssh'))
    self.assertEqual('ssh://example.com/srv/git/bar', rcp_location_to_url('example.com:/srv/git/bar'))