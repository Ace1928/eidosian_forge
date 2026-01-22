import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
class _VersionHacks(object):
    """A container to abstract the list of version hacks.

    This could be done as simply a dictionary but is abstracted like this to
    make for easier testing.
    """

    def __init__(self):
        self._discovery_data = {}

    def add_discover_hack(self, service_type, old, new=''):
        """Add a new hack for a service type.

        :param str service_type: The service_type in the catalog.
        :param re.RegexObject old: The pattern to use.
        :param str new: What to replace the pattern with.
        """
        hacks = self._discovery_data.setdefault(service_type, [])
        hacks.append((old, new))

    def get_discover_hack(self, service_type, url):
        """Apply the catalog hacks and figure out an unversioned endpoint.

        :param str service_type: the service_type to look up.
        :param str url: The original url that came from a service_catalog.

        :returns: Either the unversioned url or the one from the catalog
                  to try.
        """
        for old, new in self._discovery_data.get(service_type, []):
            new_string, number_of_subs_made = old.subn(new, url)
            if number_of_subs_made > 0:
                return new_string
        return url