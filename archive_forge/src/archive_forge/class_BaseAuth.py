import abc
import json
from keystoneauth1 import _utils as utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.identity import base
class BaseAuth(base.BaseIdentityPlugin, metaclass=abc.ABCMeta):
    """Identity V3 Authentication Plugin.

    :param string auth_url: Identity service endpoint for authentication.
    :param string trust_id: Trust ID for trust scoping.
    :param string system_scope: System information to scope to.
    :param string domain_id: Domain ID for domain scoping.
    :param string domain_name: Domain name for domain scoping.
    :param string project_id: Project ID for project scoping.
    :param string project_name: Project name for project scoping.
    :param string project_domain_id: Project's domain ID for project.
    :param string project_domain_name: Project's domain name for project.
    :param bool reauthenticate: Allow fetching a new token if the current one
                                is going to expire. (optional) default True
    :param bool include_catalog: Include the service catalog in the returned
                                 token. (optional) default True.
    """

    def __init__(self, auth_url, trust_id=None, system_scope=None, domain_id=None, domain_name=None, project_id=None, project_name=None, project_domain_id=None, project_domain_name=None, reauthenticate=True, include_catalog=True):
        super(BaseAuth, self).__init__(auth_url=auth_url, reauthenticate=reauthenticate)
        self.trust_id = trust_id
        self.system_scope = system_scope
        self.domain_id = domain_id
        self.domain_name = domain_name
        self.project_id = project_id
        self.project_name = project_name
        self.project_domain_id = project_domain_id
        self.project_domain_name = project_domain_name
        self.include_catalog = include_catalog

    @property
    def token_url(self):
        """The full URL where we will send authentication data."""
        return '%s/auth/tokens' % self.auth_url.rstrip('/')

    @abc.abstractmethod
    def get_auth_ref(self, session, **kwargs):
        return None

    @property
    def has_scope_parameters(self):
        """Return true if parameters can be used to create a scoped token."""
        return self.domain_id or self.domain_name or self.project_id or self.project_name or self.trust_id or self.system_scope