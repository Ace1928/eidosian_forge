from keystoneauth1.exceptions import auth_plugins
class OidcAccessTokenEndpointNotFound(auth_plugins.AuthPluginException):
    message = 'OpenID Connect access token endpoint not provided.'