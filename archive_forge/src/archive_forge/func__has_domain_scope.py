import abc
import urllib.parse
from keystoneauth1 import _utils as utils
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1.identity import base
@property
def _has_domain_scope(self):
    """Are there domain parameters.

        Domain parameters are v3 only so returns if any are set.

        :returns: True if a domain parameter is set, false otherwise.
        """
    return any([self._domain_id, self._domain_name, self._project_domain_id, self._project_domain_name])