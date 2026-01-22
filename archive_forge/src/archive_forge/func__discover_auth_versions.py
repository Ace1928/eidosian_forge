import urllib.parse
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ka_exc
from keystoneauth1.identity import v2 as v2_auth
from keystoneauth1.identity import v3 as v3_auth
from keystoneauth1 import session
from zaqarclient.auth import base
from zaqarclient import errors
def _discover_auth_versions(self, session, auth_url):
    v2_auth_url = None
    v3_auth_url = None
    try:
        ks_discover = discover.Discover(session=session, url=auth_url)
        v2_auth_url = ks_discover.url_for('2.0')
        v3_auth_url = ks_discover.url_for('3.0')
    except ka_exc.DiscoveryFailure:
        raise
    except ka_exc.ClientException:
        url_parts = urllib.parse.urlparse(auth_url)
        scheme, netloc, path, params, query, fragment = url_parts
        path = path.lower()
        if path.startswith('/v3'):
            v3_auth_url = auth_url
        elif path.startswith('/v2'):
            v2_auth_url = auth_url
        else:
            raise errors.ZaqarError('Unable to determine the Keystone version to authenticate with using the given auth_url.')
    return (v2_auth_url, v3_auth_url)