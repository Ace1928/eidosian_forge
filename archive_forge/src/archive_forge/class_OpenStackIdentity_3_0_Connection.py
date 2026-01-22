import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackIdentity_3_0_Connection(OpenStackIdentityConnection):
    """
    Connection class for Keystone API v3.x.
    """
    responseCls = OpenStackAuthResponse
    name = 'OpenStack Identity API v3.x'
    auth_version = '3.0'
    VALID_TOKEN_SCOPES = [OpenStackIdentityTokenScope.PROJECT, OpenStackIdentityTokenScope.DOMAIN, OpenStackIdentityTokenScope.UNSCOPED]

    def __init__(self, auth_url, user_id, key, tenant_name=None, domain_name='Default', tenant_domain_id='default', token_scope=OpenStackIdentityTokenScope.PROJECT, timeout=None, proxy_url=None, parent_conn=None, auth_cache=None):
        """
        :param tenant_name: Name of the project this user belongs to. Note:
                            When token_scope is set to project, this argument
                            control to which project to scope the token to.
        :type tenant_name: ``str``

        :param domain_name: Domain the user belongs to. Note: When token_scope
                            is set to token, this argument controls to which
                            domain to scope the token to.
        :type domain_name: ``str``

        :param token_scope: Whether to scope a token to a "project", a
                            "domain" or "unscoped"
        :type token_scope: ``str``

        :param auth_cache: Where to cache authentication tokens.
        :type auth_cache: :class:`OpenStackAuthenticationCache`
        """
        super().__init__(auth_url=auth_url, user_id=user_id, key=key, tenant_name=tenant_name, domain_name=domain_name, tenant_domain_id=tenant_domain_id, token_scope=token_scope, timeout=timeout, proxy_url=proxy_url, parent_conn=parent_conn, auth_cache=auth_cache)
        if self.token_scope not in self.VALID_TOKEN_SCOPES:
            raise ValueError('Invalid value for "token_scope" argument: %s' % self.token_scope)
        if self.token_scope == OpenStackIdentityTokenScope.PROJECT and (not self.tenant_name or not self.domain_name):
            raise ValueError('Must provide tenant_name and domain_name argument')
        elif self.token_scope == OpenStackIdentityTokenScope.DOMAIN and (not self.domain_name):
            raise ValueError('Must provide domain_name argument')

    def authenticate(self, force=False):
        """
        Perform authentication.
        """
        if not self._is_authentication_needed(force=force):
            return self
        data = self._get_auth_data()
        data = json.dumps(data)
        response = self.request('/v3/auth/tokens', data=data, headers={'Content-Type': 'application/json'}, method='POST')
        self._parse_token_response(response, cache_it=True)
        return self

    def list_domains(self):
        """
        List the available domains.

        :rtype: ``list`` of :class:`OpenStackIdentityDomain`
        """
        response = self.authenticated_request('/v3/domains', method='GET')
        result = self._to_domains(data=response.object['domains'])
        return result

    def list_projects(self):
        """
        List the available projects.

        Note: To perform this action, user you are currently authenticated with
        needs to be an admin.

        :rtype: ``list`` of :class:`OpenStackIdentityProject`
        """
        response = self.authenticated_request('/v3/projects', method='GET')
        result = self._to_projects(data=response.object['projects'])
        return result

    def list_users(self):
        """
        List the available users.

        :rtype: ``list`` of :class:`.OpenStackIdentityUser`
        """
        response = self.authenticated_request('/v3/users', method='GET')
        result = self._to_users(data=response.object['users'])
        return result

    def list_roles(self):
        """
        List the available roles.

        :rtype: ``list`` of :class:`.OpenStackIdentityRole`
        """
        response = self.authenticated_request('/v3/roles', method='GET')
        result = self._to_roles(data=response.object['roles'])
        return result

    def get_domain(self, domain_id):
        """
        Retrieve information about a single domain.

        :param domain_id: ID of domain to retrieve information for.
        :type domain_id: ``str``

        :rtype: :class:`.OpenStackIdentityDomain`
        """
        response = self.authenticated_request('/v3/domains/%s' % domain_id, method='GET')
        result = self._to_domain(data=response.object['domain'])
        return result

    def get_user(self, user_id):
        """
        Get a user account by ID.

        :param user_id: User's id.
        :type name: ``str``

        :return: Located user.
        :rtype: :class:`.OpenStackIdentityUser`
        """
        response = self.authenticated_request('/v3/users/%s' % user_id)
        user = self._to_user(data=response.object['user'])
        return user

    def list_user_projects(self, user):
        """
        Retrieve all the projects user belongs to.

        :rtype: ``list`` of :class:`.OpenStackIdentityProject`
        """
        path = '/v3/users/%s/projects' % user.id
        response = self.authenticated_request(path, method='GET')
        result = self._to_projects(data=response.object['projects'])
        return result

    def list_user_domain_roles(self, domain, user):
        """
        Retrieve all the roles for a particular user on a domain.

        :rtype: ``list`` of :class:`.OpenStackIdentityRole`
        """
        path = '/v3/domains/{}/users/{}/roles'.format(domain.id, user.id)
        response = self.authenticated_request(path, method='GET')
        result = self._to_roles(data=response.object['roles'])
        return result

    def grant_domain_role_to_user(self, domain, role, user):
        """
        Grant domain role to a user.

        Note: This function appears to be idempotent.

        :param domain: Domain to grant the role to.
        :type domain: :class:`.OpenStackIdentityDomain`

        :param role: Role to grant.
        :type role: :class:`.OpenStackIdentityRole`

        :param user: User to grant the role to.
        :type user: :class:`.OpenStackIdentityUser`

        :return: ``True`` on success.
        :rtype: ``bool``
        """
        path = '/v3/domains/{}/users/{}/roles/{}'.format(domain.id, user.id, role.id)
        response = self.authenticated_request(path, method='PUT')
        return response.status == httplib.NO_CONTENT

    def revoke_domain_role_from_user(self, domain, user, role):
        """
        Revoke domain role from a user.

        :param domain: Domain to revoke the role from.
        :type domain: :class:`.OpenStackIdentityDomain`

        :param role: Role to revoke.
        :type role: :class:`.OpenStackIdentityRole`

        :param user: User to revoke the role from.
        :type user: :class:`.OpenStackIdentityUser`

        :return: ``True`` on success.
        :rtype: ``bool``
        """
        path = '/v3/domains/{}/users/{}/roles/{}'.format(domain.id, user.id, role.id)
        response = self.authenticated_request(path, method='DELETE')
        return response.status == httplib.NO_CONTENT

    def grant_project_role_to_user(self, project, role, user):
        """
        Grant project role to a user.

        Note: This function appears to be idempotent.

        :param project: Project to grant the role to.
        :type project: :class:`.OpenStackIdentityDomain`

        :param role: Role to grant.
        :type role: :class:`.OpenStackIdentityRole`

        :param user: User to grant the role to.
        :type user: :class:`.OpenStackIdentityUser`

        :return: ``True`` on success.
        :rtype: ``bool``
        """
        path = '/v3/projects/{}/users/{}/roles/{}'.format(project.id, user.id, role.id)
        response = self.authenticated_request(path, method='PUT')
        return response.status == httplib.NO_CONTENT

    def revoke_project_role_from_user(self, project, role, user):
        """
        Revoke project role from a user.

        :param project: Project to revoke the role from.
        :type project: :class:`.OpenStackIdentityDomain`

        :param role: Role to revoke.
        :type role: :class:`.OpenStackIdentityRole`

        :param user: User to revoke the role from.
        :type user: :class:`.OpenStackIdentityUser`

        :return: ``True`` on success.
        :rtype: ``bool``
        """
        path = '/v3/projects/{}/users/{}/roles/{}'.format(project.id, user.id, role.id)
        response = self.authenticated_request(path, method='DELETE')
        return response.status == httplib.NO_CONTENT

    def create_user(self, email, password, name, description=None, domain_id=None, default_project_id=None, enabled=True):
        """
        Create a new user account.

        :param email: User's mail address.
        :type email: ``str``

        :param password: User's password.
        :type password: ``str``

        :param name: User's name.
        :type name: ``str``

        :param description: Optional description.
        :type description: ``str``

        :param domain_id: ID of the domain to add the user to (optional).
        :type domain_id: ``str``

        :param default_project_id: ID of the default user project (optional).
        :type default_project_id: ``str``

        :param enabled: True to enable user after creation.
        :type enabled: ``bool``

        :return: Created user.
        :rtype: :class:`.OpenStackIdentityUser`
        """
        data = {'email': email, 'password': password, 'name': name, 'enabled': enabled}
        if description:
            data['description'] = description
        if domain_id:
            data['domain_id'] = domain_id
        if default_project_id:
            data['default_project_id'] = default_project_id
        data = json.dumps({'user': data})
        response = self.authenticated_request('/v3/users', data=data, method='POST')
        user = self._to_user(data=response.object['user'])
        return user

    def enable_user(self, user):
        """
        Enable user account.

        Note: This operation appears to be idempotent.

        :param user: User to enable.
        :type user: :class:`.OpenStackIdentityUser`

        :return: User account which has been enabled.
        :rtype: :class:`.OpenStackIdentityUser`
        """
        data = {'enabled': True}
        data = json.dumps({'user': data})
        response = self.authenticated_request('/v3/users/%s' % user.id, data=data, method='PATCH')
        user = self._to_user(data=response.object['user'])
        return user

    def disable_user(self, user):
        """
        Disable user account.

        Note: This operation appears to be idempotent.

        :param user: User to disable.
        :type user: :class:`.OpenStackIdentityUser`

        :return: User account which has been disabled.
        :rtype: :class:`.OpenStackIdentityUser`
        """
        data = {'enabled': False}
        data = json.dumps({'user': data})
        response = self.authenticated_request('/v3/users/%s' % user.id, data=data, method='PATCH')
        user = self._to_user(data=response.object['user'])
        return user

    def _get_auth_data(self):
        data = {'auth': {'identity': {'methods': ['password'], 'password': {'user': {'domain': {'name': self.domain_name}, 'name': self.user_id, 'password': self.key}}}}}
        if self.token_scope == OpenStackIdentityTokenScope.PROJECT:
            data['auth']['scope'] = {'project': {'domain': {'id': self.tenant_domain_id}, 'name': self.tenant_name}}
        elif self.token_scope == OpenStackIdentityTokenScope.DOMAIN:
            data['auth']['scope'] = {'domain': {'name': self.domain_name}}
        elif self.token_scope == OpenStackIdentityTokenScope.UNSCOPED:
            pass
        else:
            raise ValueError('Token needs to be scoped either to project or a domain')
        return data

    def _load_auth_context_from_cache(self):
        context = super()._load_auth_context_from_cache()
        if context is None:
            return None
        try:
            self._fetch_auth_token()
        except InvalidCredsError:
            return None
        return context

    def _parse_token_response(self, response, cache_it=False, raise_ambiguous_version_error=True):
        """
        Parse a response from /v3/auth/tokens.

        :param cache_it: Should we cache the authentication context?
        :type cache_it: ``bool``

        :param raise_ambiguous_version_error: Should an ambiguous version
            error be raised on a 300 response?
        :type raise_ambiguous_version_error: ``bool``
        """
        if response.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError()
        elif response.status in [httplib.OK, httplib.CREATED]:
            headers = response.headers
            try:
                body = json.loads(response.body)
            except Exception as e:
                raise MalformedResponseError('Failed to parse JSON', e)
            try:
                roles = self._to_roles(body['token']['roles'])
            except Exception:
                roles = []
            try:
                expires = parse_date(body['token']['expires_at'])
                token = headers['x-subject-token']
                if cache_it:
                    self._cache_auth_context(OpenStackAuthenticationContext(token, expiration=expires))
                self.auth_token = token
                self.auth_token_expires = expires
                self.urls = body['token'].get('catalog', None)
                self.auth_user_info = body['token'].get('user', None)
                self.auth_user_roles = roles
            except KeyError as e:
                raise MalformedResponseError('Auth JSON response is                                              missing required elements', e)
        elif raise_ambiguous_version_error and response.status == 300:
            raise LibcloudError('Auth request returned ambiguous version error, tryusing the version specific URL to connect, e.g. identity/v3/auth/tokens')
        else:
            body = 'code: {} body:{}'.format(response.status, response.body)
            raise MalformedResponseError('Malformed response', body=body, driver=self.driver)

    def _fetch_auth_token(self):
        """
        Fetch our authentication token and service catalog.
        """
        headers = {'X-Subject-Token': self.auth_token}
        response = self.authenticated_request('/v3/auth/tokens', headers=headers)
        self._parse_token_response(response)
        return self

    def _to_domains(self, data):
        result = []
        for item in data:
            domain = self._to_domain(data=item)
            result.append(domain)
        return result

    def _to_domain(self, data):
        domain = OpenStackIdentityDomain(id=data['id'], name=data['name'], enabled=data['enabled'])
        return domain

    def _to_users(self, data):
        result = []
        for item in data:
            user = self._to_user(data=item)
            result.append(user)
        return result

    def _to_user(self, data):
        user = OpenStackIdentityUser(id=data['id'], domain_id=data['domain_id'], name=data['name'], email=data.get('email'), description=data.get('description', None), enabled=data.get('enabled'))
        return user

    def _to_roles(self, data):
        result = []
        for item in data:
            user = self._to_role(data=item)
            result.append(user)
        return result

    def _to_role(self, data):
        role = OpenStackIdentityRole(id=data['id'], name=data['name'], description=data.get('description', None), enabled=data.get('enabled', True))
        return role