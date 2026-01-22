from keystoneauth1.exceptions import base
class MissingAuthPlugin(AuthPluginException):
    message = 'An authenticated request is required but no plugin available.'