from keystoneauth1.exceptions import auth_plugins
class InvalidDiscoveryEndpoint(auth_plugins.AuthPluginException):
    message = 'OpenID Connect Discovery Document endpoint not set.'