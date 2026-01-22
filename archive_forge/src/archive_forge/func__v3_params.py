import abc
import urllib.parse
from keystoneauth1 import _utils as utils
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1.identity import base
@property
def _v3_params(self):
    """Return the parameters that are common to v3 plugins."""
    return {'trust_id': self._trust_id, 'system_scope': self._system_scope, 'project_id': self._project_id, 'project_name': self._project_name, 'project_domain_id': self.project_domain_id, 'project_domain_name': self.project_domain_name, 'domain_id': self._domain_id, 'domain_name': self._domain_name, 'reauthenticate': self.reauthenticate}