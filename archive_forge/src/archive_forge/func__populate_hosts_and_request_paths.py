from libcloud.utils.py3 import ET, httplib
from libcloud.common.base import Response, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.types import LibcloudError, MalformedResponseError, KeyPairDoesNotExistError
from libcloud.common.exceptions import BaseHTTPError
from libcloud.common.openstack_identity import (
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