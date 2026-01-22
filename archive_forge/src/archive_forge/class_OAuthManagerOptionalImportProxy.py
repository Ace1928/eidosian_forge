from keystoneclient.i18n import _
from keystoneclient.v3.contrib.oauth1 import access_tokens
from keystoneclient.v3.contrib.oauth1 import consumers
from keystoneclient.v3.contrib.oauth1 import request_tokens
class OAuthManagerOptionalImportProxy(object):
    """Act as a proxy manager in case oauthlib is no installed.

    This class will only be created if oauthlib is not in the system,
    trying to access any of the attributes in name (access_tokens,
    consumers, request_tokens), will result in a NotImplementedError,
    and a message.

    >>> manager.access_tokens.blah
    NotImplementedError: To use 'access_tokens' oauthlib must be installed

    Otherwise, if trying to access an attribute other than the ones in name,
    the manager will state that the attribute does not exist.

    >>> manager.dne.blah
    AttributeError: 'OAuthManagerOptionalImportProxy' object has no
    attribute 'dne'
    """

    def __getattribute__(self, name):
        """Return error when name is related to oauthlib and not exist."""
        if name in ('access_tokens', 'consumers', 'request_tokens'):
            raise NotImplementedError(_('To use %r oauthlib must be installed') % name)
        return super(OAuthManagerOptionalImportProxy, self).__getattribute__(name)