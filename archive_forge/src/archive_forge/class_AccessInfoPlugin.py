from keystoneauth1.identity import base
class AccessInfoPlugin(base.BaseIdentityPlugin):
    """A plugin that turns an existing AccessInfo object into a usable plugin.

    There are cases where reuse of an auth_ref or AccessInfo object is
    warranted such as from a cache, from auth_token middleware, or another
    source.

    Turn the existing access info object into an identity plugin. This plugin
    cannot be refreshed as the AccessInfo object does not contain any
    authorizing information.

    :param auth_ref: the existing AccessInfo object.
    :type auth_ref: keystoneauth1.access.AccessInfo
    :param auth_url: the url where this AccessInfo was retrieved from. Required
                     if using the AUTH_INTERFACE with get_endpoint. (optional)
    """

    def __init__(self, auth_ref, auth_url=None):
        super(AccessInfoPlugin, self).__init__(auth_url=auth_url, reauthenticate=False)
        self.auth_ref = auth_ref

    def get_auth_ref(self, session, **kwargs):
        return self.auth_ref

    def invalidate(self):
        return False