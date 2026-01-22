import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def _get_catalog_discover_hack(self):
    """Apply the catalog hacks and figure out an unversioned endpoint.

        This function is internal to keystoneauth1.

        :returns: A url that has been transformed by the regex hacks that
                  match the service_type.
        """
    return _VERSION_HACKS.get_discover_hack(self.service_type, self.url)