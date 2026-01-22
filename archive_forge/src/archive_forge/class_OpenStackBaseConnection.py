from libcloud.utils.py3 import ET, httplib
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.types import LibcloudError, MalformedResponseError, KeyPairDoesNotExistError
from libcloud.common.exceptions import BaseHTTPError
from libcloud.common.openstack_identity import (
class OpenStackBaseConnection(ConnectionUserAndKey):
    """
    Base class for OpenStack connections.

    :param user_id: User name to use when authenticating
    :type user_id: ``str``

    :param key: Secret to use when authenticating.
    :type key: ``str``

    :param secure: Use HTTPS?  (True by default.)
    :type secure: ``bool``

    :param ex_force_base_url: Base URL for connection requests.  If
                              not specified, this will be determined by
                              authenticating.
    :type ex_force_base_url: ``str``

    :param ex_force_auth_url: Base URL for authentication requests.
    :type ex_force_auth_url: ``str``

    :param ex_force_auth_version: Authentication version to use.  If
                                  not specified, defaults to AUTH_API_VERSION.
    :type ex_force_auth_version: ``str``

    :param ex_force_auth_token: Authentication token to use for connection
                                requests.  If specified, the connection will
                                not attempt to authenticate, and the value
                                of ex_force_base_url will be used to
                                determine the base request URL.  If
                                ex_force_auth_token is passed in,
                                ex_force_base_url must also be provided.
    :type ex_force_auth_token: ``str``

    :param token_scope: Whether to scope a token to a "project", a
                        "domain" or "unscoped".
    :type token_scope: ``str``

    :param ex_domain_name: When authenticating, provide this domain name to
                           the identity service.  A scoped token will be
                           returned. Some cloud providers require the domain
                           name to be provided at authentication time. Others
                           will use a default domain if none is provided.
    :type ex_domain_name: ``str``

    :param ex_tenant_name: When authenticating, provide this tenant name to the
                           identity service. A scoped token will be returned.
                           Some cloud providers require the tenant name to be
                           provided at authentication time. Others will use a
                           default tenant if none is provided.
    :type ex_tenant_name: ``str``

    :param ex_tenant_domain_id: When authenticating, provide this tenant
                                domain id to the identity service.
                                A scoped token will be returned.
                                Some cloud providers require the tenant
                                domain id to be provided at authentication
                                time. Others will use a default tenant
                                domain id if none is provided.
    :type ex_tenant_domain_id: ``str``

    :param ex_force_service_type: Service type to use when selecting an
                                  service. If not specified, a provider
                                  specific default will be used.
    :type ex_force_service_type: ``str``

    :param ex_force_service_name: Service name to use when selecting an
                                  service. If not specified, a provider
                                  specific default will be used.
    :type ex_force_service_name: ``str``

    :param ex_force_service_region: Region to use when selecting an service.
                                    If not specified, a provider specific
                                    default will be used.
    :type ex_force_service_region: ``str``

    :param ex_auth_cache: External cache where authentication tokens are
                          stored for reuse by other processes. Tokens are
                          always cached in memory on the driver instance. To
                          share tokens among multiple drivers, processes, or
                          systems, pass a cache here.
    :type ex_auth_cache: :class:`OpenStackAuthenticationCache`
    """
    auth_url = None
    auth_token = None
    auth_token_expires = None
    auth_user_info = None
    service_catalog = None
    service_type = None
    service_name = None
    service_region = None
    accept_format = None
    _auth_version = None

    def __init__(self, user_id, key, secure=True, host=None, port=None, timeout=None, proxy_url=None, ex_force_base_url=None, ex_force_auth_url=None, ex_force_auth_version=None, ex_force_auth_token=None, ex_token_scope=OpenStackIdentityTokenScope.PROJECT, ex_domain_name='Default', ex_tenant_name=None, ex_tenant_domain_id='default', ex_force_service_type=None, ex_force_service_name=None, ex_force_service_region=None, ex_force_microversion=None, ex_auth_cache=None, retry_delay=None, backoff=None):
        super().__init__(user_id, key, secure=secure, timeout=timeout, retry_delay=retry_delay, backoff=backoff, proxy_url=proxy_url)
        if ex_force_auth_version:
            self._auth_version = ex_force_auth_version
        self.base_url = ex_force_base_url
        self._ex_force_base_url = ex_force_base_url
        self._ex_force_auth_url = ex_force_auth_url
        self._ex_force_auth_token = ex_force_auth_token
        self._ex_token_scope = ex_token_scope
        self._ex_domain_name = ex_domain_name
        self._ex_tenant_name = ex_tenant_name
        self._ex_tenant_domain_id = ex_tenant_domain_id
        self._ex_force_service_type = ex_force_service_type
        self._ex_force_service_name = ex_force_service_name
        self._ex_force_service_region = ex_force_service_region
        self._ex_force_microversion = ex_force_microversion
        self._ex_auth_cache = ex_auth_cache
        self._osa = None
        if ex_force_auth_token and (not ex_force_base_url):
            raise LibcloudError('Must also provide ex_force_base_url when specifying ex_force_auth_token.')
        if ex_force_auth_token:
            self.auth_token = ex_force_auth_token
        if not self._auth_version:
            self._auth_version = AUTH_API_VERSION
        auth_url = self._get_auth_url()
        if not auth_url:
            raise LibcloudError('OpenStack instance must ' + 'have auth_url set')

    def get_auth_class(self):
        """
        Retrieve identity / authentication class instance.

        :rtype: :class:`OpenStackIdentityConnection`
        """
        if not self._osa:
            auth_url = self._get_auth_url()
            cls = get_class_for_auth_version(auth_version=self._auth_version)
            self._osa = cls(auth_url=auth_url, user_id=self.user_id, key=self.key, tenant_name=self._ex_tenant_name, tenant_domain_id=self._ex_tenant_domain_id, domain_name=self._ex_domain_name, token_scope=self._ex_token_scope, timeout=self.timeout, proxy_url=self.proxy_url, parent_conn=self, auth_cache=self._ex_auth_cache)
        return self._osa

    def request(self, action, params=None, data='', headers=None, method='GET', raw=False):
        headers = headers or {}
        params = params or {}
        default_content_type = getattr(self, 'default_content_type', None)
        if method.upper() in ['POST', 'PUT'] and default_content_type:
            headers['Content-Type'] = default_content_type
        try:
            return super().request(action=action, params=params, data=data, method=method, headers=headers, raw=raw)
        except BaseHTTPError as ex:
            if ex.code == httplib.UNAUTHORIZED and self._ex_force_auth_token is None:
                self.get_auth_class().clear_cached_auth_context()
            raise

    def _get_auth_url(self):
        """
        Retrieve auth url for this instance using either "ex_force_auth_url"
        constructor kwarg of "auth_url" class variable.
        """
        auth_url = self.auth_url
        if self._ex_force_auth_url is not None:
            auth_url = self._ex_force_auth_url
        return auth_url

    def get_service_catalog(self):
        if self.service_catalog is None:
            self._populate_hosts_and_request_paths()
        return self.service_catalog

    def get_service_name(self):
        """
        Gets the service name used to look up the endpoint in the service
        catalog.

        :return: name of the service in the catalog
        """
        if self._ex_force_service_name:
            return self._ex_force_service_name
        return self.service_name

    def get_endpoint(self):
        """
        Selects the endpoint to use based on provider specific values,
        or overrides passed in by the user when setting up the driver.

        :returns: url of the relevant endpoint for the driver
        """
        service_type = self.service_type
        service_name = self.service_name
        service_region = self.service_region
        if self._ex_force_service_type:
            service_type = self._ex_force_service_type
        if self._ex_force_service_name:
            service_name = self._ex_force_service_name
        if self._ex_force_service_region:
            service_region = self._ex_force_service_region
        endpoint = self.service_catalog.get_endpoint(service_type=service_type, name=service_name, region=service_region)
        url = endpoint.url
        if not url:
            raise LibcloudError('Could not find specified endpoint')
        return url

    def add_default_headers(self, headers):
        headers[AUTH_TOKEN_HEADER] = self.auth_token
        headers['Accept'] = self.accept_format
        if self._ex_force_microversion:
            microversion = self._ex_force_microversion.strip().split()
            if len(microversion) == 2:
                service_type = microversion[0]
                microversion = microversion[1]
            elif len(microversion) == 1:
                service_type = 'compute'
                microversion = microversion[0]
            else:
                raise LibcloudError('Invalid microversion format: servicename X.XX')
            if self.service_type and self.service_type.startswith(service_type):
                headers['OpenStack-API-Version'] = '{} {}'.format(service_type, microversion)
        return headers

    def morph_action_hook(self, action):
        self._populate_hosts_and_request_paths()
        return super().morph_action_hook(action)

    def _set_up_connection_info(self, url):
        prev_conn = (self.host, self.port, self.secure)
        result = self._tuple_from_url(url)
        self.host, self.port, self.secure, self.request_path = result
        new_conn = (self.host, self.port, self.secure)
        if new_conn != prev_conn:
            self.connect()

    def _populate_hosts_and_request_paths(self):
        """
        OpenStack uses a separate host for API calls which is only provided
        after an initial authentication request.
        """
        osa = self.get_auth_class()
        if self._ex_force_auth_token:
            self._set_up_connection_info(url=self._ex_force_base_url)
            return
        if not osa.is_token_valid():
            if self._auth_version == '2.0_apikey':
                kwargs = {'auth_type': 'api_key'}
            elif self._auth_version == '2.0_password':
                kwargs = {'auth_type': 'password'}
            else:
                kwargs = {}
            osa = osa.authenticate(**kwargs)
            self.auth_token = osa.auth_token
            self.auth_token_expires = osa.auth_token_expires
            self.auth_user_info = osa.auth_user_info
            osc = OpenStackServiceCatalog(service_catalog=osa.urls, auth_version=self._auth_version)
            self.service_catalog = osc
        url = self._ex_force_base_url or self.get_endpoint()
        self._set_up_connection_info(url=url)