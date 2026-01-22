import errno
import os
import warnings
from lazr.restfulclient.resource import (  # noqa: F401
from lazr.restfulclient.authorize.oauth import SystemWideConsumer
from lazr.restfulclient._browser import RestfulHttp
from launchpadlib.credentials import (
from launchpadlib import uris
from launchpadlib.uris import (  # noqa: F401
class PillarSet(CollectionWithKeyBasedLookup):
    """A custom subclass capable of lookup by pillar name.

    Projects, project groups, and distributions are all pillars.
    """

    def _get_url_from_id(self, key):
        """Transform a project name into the URL to a project resource."""
        return str(self._root._root_uri.ensureSlash()) + str(key)
    collection_of = None