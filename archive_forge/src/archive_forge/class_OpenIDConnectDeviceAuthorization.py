from keystoneauth1 import exceptions
from keystoneauth1 import identity
from keystoneauth1 import loading
class OpenIDConnectDeviceAuthorization(_OpenIDConnectBase):

    @property
    def plugin_class(self):
        return identity.V3OidcDeviceAuthorization

    def get_options(self):
        options = super(OpenIDConnectDeviceAuthorization, self).get_options()
        options = [opt for opt in options if opt.name != 'access-token-type']
        options.extend([loading.Opt('device-authorization-endpoint', help='OAuth 2.0 Device Authorization Endpoint. Note that if a discovery document is being passed this option will override the endpoint provided by the server in the discovery document.'), loading.Opt('code-challenge-method', help='PKCE Challenge Method (RFC 7636)')])
        return options