import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def versioned_url_for(self, min_version=None, max_version=None, **kwargs):
    """Get the endpoint url for a version.

        min_version and max_version can be given either as strings or tuples.

        :param min_version: The minimum version that is acceptable. If
            min_version is given with no max_version it is as if max version
            is 'latest'.
        :param max_version: The maximum version that is acceptable. If
            min_version is given with no max_version it is as if max version is
            'latest'.

        :returns: The url for the specified version or None if no match.
        :rtype: str
        """
    data = self.versioned_data_for(min_version=min_version, max_version=max_version, **kwargs)
    return data['url'] if data else None