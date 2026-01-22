import urllib.parse
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ka_exc
from keystoneauth1.identity import v2 as v2_auth
from keystoneauth1.identity import v3 as v3_auth
from keystoneauth1 import session
from zaqarclient.auth import base
from zaqarclient import errors
def _get_endpoint(self, ks_session, **kwargs):
    """Get an endpoint using the provided keystone session."""
    endpoint_type = kwargs.get('endpoint_type') or 'publicURL'
    service_type = kwargs.get('service_type') or 'messaging'
    region_name = kwargs.get('region_name')
    endpoint = ks_session.get_endpoint(service_type=service_type, interface=endpoint_type, region_name=region_name)
    return endpoint