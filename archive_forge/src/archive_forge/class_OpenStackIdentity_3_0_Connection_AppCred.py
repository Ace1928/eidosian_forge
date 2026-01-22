import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackIdentity_3_0_Connection_AppCred(OpenStackIdentity_3_0_Connection):
    """
    Connection class for Keystone API v3.x using Application Credentials.

    'user_id' is the application credential id and 'key' is the application
    credential secret.
    """
    name = 'OpenStack Identity API v3.x with Application Credentials'

    def __init__(self, auth_url, user_id, key, tenant_name=None, domain_name=None, tenant_domain_id=None, token_scope=None, timeout=None, proxy_url=None, parent_conn=None, auth_cache=None):
        """
        Tenant, domain and scope options are ignored as they are contained
        within the app credential itself and can't be changed.
        """
        super().__init__(auth_url=auth_url, user_id=user_id, key=key, tenant_name=tenant_name, domain_name=domain_name, token_scope=OpenStackIdentityTokenScope.UNSCOPED, timeout=timeout, proxy_url=proxy_url, parent_conn=parent_conn, auth_cache=auth_cache)

    def _get_auth_data(self):
        data = {'auth': {'identity': {'methods': ['application_credential'], 'application_credential': {'id': self.user_id, 'secret': self.key}}}}
        return data