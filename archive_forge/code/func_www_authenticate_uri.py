import urllib.parse
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import plugin
from keystoneclient.v3 import client as v3_client
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.i18n import _
@property
def www_authenticate_uri(self):
    www_authenticate_uri = self._adapter.get_endpoint(interface=plugin.AUTH_INTERFACE)
    if isinstance(self._adapter.auth, _auth.AuthTokenPlugin):
        www_authenticate_uri = urllib.parse.urljoin(www_authenticate_uri, '/').rstrip('/')
    return www_authenticate_uri