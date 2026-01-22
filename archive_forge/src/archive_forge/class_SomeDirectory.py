from .. import osutils, tests, urlutils
from ..directory_service import directories
from ..location import hooks as location_hooks
from ..location import location_to_url, rcp_location_to_url
class SomeDirectory:

    def look_up(self, name, url, purpose=None):
        return 'http://bar'