from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneclient import base
from keystoneclient.v3 import domains
from keystoneclient.v3 import projects
from keystoneclient.v3 import system
class AuthManager(base.Manager):
    """Retrieve auth context specific information.

    The information returned by the auth routes is entirely dependent on the
    authentication information provided by the user.
    """
    _PROJECTS_URL = '/auth/projects'
    _DOMAINS_URL = '/auth/domains'
    _SYSTEM_URL = '/auth/system'

    def projects(self):
        """List projects that the specified token can be rescoped to.

        :returns: a list of projects.
        :rtype: list of :class:`keystoneclient.v3.projects.Project`

        """
        try:
            return self._list(self._PROJECTS_URL, 'projects', obj_class=Project)
        except exceptions.EndpointNotFound:
            endpoint_filter = {'interface': plugin.AUTH_INTERFACE}
            return self._list(self._PROJECTS_URL, 'projects', obj_class=Project, endpoint_filter=endpoint_filter)

    def domains(self):
        """List Domains that the specified token can be rescoped to.

        :returns: a list of domains.
        :rtype: list of :class:`keystoneclient.v3.domains.Domain`.

        """
        try:
            return self._list(self._DOMAINS_URL, 'domains', obj_class=Domain)
        except exceptions.EndpointNotFound:
            endpoint_filter = {'interface': plugin.AUTH_INTERFACE}
            return self._list(self._DOMAINS_URL, 'domains', obj_class=Domain, endpoint_filter=endpoint_filter)

    def systems(self):
        """List Systems that the specified token can be rescoped to.

        At the moment this is either empty or "all".

        :returns: a list of systems.
        :rtype: list of :class:`keystoneclient.v3.systems.System`.

        """
        try:
            return self._list(self._SYSTEM_URL, 'system', obj_class=System)
        except exceptions.EndpointNotFound:
            endpoint_filter = {'interface': plugin.AUTH_INTERFACE}
            return self._list(self._SYSTEM_URL, 'system', obj_class=System, endpoint_filter=endpoint_filter)