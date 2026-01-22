from oslo_config import cfg
from webob import exc
from heat.common import endpoint_utils
from heat.common.i18n import _
from heat.common import wsgi
def _validate_auth_url(self, auth_url):
    """Validate auth_url to ensure it can be used."""
    if not auth_url:
        raise exc.HTTPBadRequest(_('Request missing required header X-Auth-Url'))
    allowed = cfg.CONF.auth_password.allowed_auth_uris
    if auth_url not in allowed:
        raise exc.HTTPUnauthorized(_('Header X-Auth-Url "%s" not an allowed endpoint') % auth_url)
    return True