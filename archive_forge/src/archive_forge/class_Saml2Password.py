from keystoneauth1.extras import _saml2
from keystoneauth1 import loading
class Saml2Password(loading.BaseFederationLoader):

    @property
    def plugin_class(self):
        return _saml2.V3Saml2Password

    @property
    def available(self):
        return _saml2._V3_SAML2_AVAILABLE

    def get_options(self):
        options = super(Saml2Password, self).get_options()
        options.extend([loading.Opt('identity-provider-url', required=True, help='An Identity Provider URL, where the SAML2 authentication request will be sent.'), loading.Opt('username', help='Username', required=True), loading.Opt('password', secret=True, help='Password', required=True)])
        return options