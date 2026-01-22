from keystoneauth1.exceptions import base
class AuthPluginException(base.ClientException):
    message = 'Unknown error with authentication plugins.'