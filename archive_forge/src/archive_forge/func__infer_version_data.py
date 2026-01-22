import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def _infer_version_data(self, project_id=None):
    """Return version data dict for when discovery fails.

        :param string project_id: ID of the currently scoped project. Used for
                                  removing project_id components of URLs from
                                  the catalog. (optional)
        :returns: A list of :class:`VersionData` sorted by version number.
        :rtype: list(VersionData)
        """
    version = self.api_version
    if version:
        version = version_to_string(self.api_version)
    url = self.url.rstrip('/')
    if project_id and url.endswith(project_id):
        url, _ = self.url.rsplit('/', 1)
    url += '/'
    return [VersionData(url=url, version=version)]