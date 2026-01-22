import abc
import urllib.parse
from keystoneauth1 import _utils as utils
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1.identity import base
def _do_create_plugin(self, session):
    plugin = None
    try:
        disc = self.get_discovery(session, self.auth_url, authenticated=False)
    except (exceptions.DiscoveryFailure, exceptions.HttpError, exceptions.SSLError, exceptions.ConnectionError) as e:
        LOG.warning('Failed to discover available identity versions when contacting %s. Attempting to parse version from URL.', self.auth_url)
        url_parts = urllib.parse.urlparse(self.auth_url)
        path = url_parts.path.lower()
        if path.startswith('/v2.0'):
            if self._has_domain_scope:
                raise exceptions.DiscoveryFailure('Cannot use v2 authentication with domain scope')
            plugin = self.create_plugin(session, (2, 0), self.auth_url)
        elif path.startswith('/v3'):
            plugin = self.create_plugin(session, (3, 0), self.auth_url)
        else:
            raise exceptions.DiscoveryFailure('Could not find versioned identity endpoints when attempting to authenticate. Please check that your auth_url is correct. %s' % e)
    else:
        reverse = self._default_domain_id or self._default_domain_name
        disc_data = disc.version_data(reverse=bool(reverse))
        v2_with_domain_scope = False
        for data in disc_data:
            version = data['version']
            if discover.version_match((2,), version) and self._has_domain_scope:
                v2_with_domain_scope = True
                continue
            plugin = self.create_plugin(session, version, data['url'], raw_status=data['raw_status'])
            if plugin:
                break
        if not plugin and v2_with_domain_scope:
            raise exceptions.DiscoveryFailure('Cannot use v2 authentication with domain scope')
    if plugin:
        return plugin
    raise exceptions.DiscoveryFailure('Could not find versioned identity endpoints when attempting to authenticate. Please check that your auth_url is correct.')