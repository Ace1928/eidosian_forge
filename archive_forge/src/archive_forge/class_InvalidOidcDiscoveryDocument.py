from keystoneauth1.exceptions import auth_plugins
class InvalidOidcDiscoveryDocument(auth_plugins.AuthPluginException):
    message = 'OpenID Connect Discovery Document is not valid JSON.'