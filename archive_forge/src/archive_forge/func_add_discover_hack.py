import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def add_discover_hack(self, service_type, old, new=''):
    """Add a new hack for a service type.

        :param str service_type: The service_type in the catalog.
        :param re.RegexObject old: The pattern to use.
        :param str new: What to replace the pattern with.
        """
    hacks = self._discovery_data.setdefault(service_type, [])
    hacks.append((old, new))