from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class UnsupportedTokenTypeError(OAuth2Error):
    """
    The authorization server does not support the hint of the
    presented token type.  I.e. the client tried to revoke an access token
    on a server not supporting this feature.
    """
    error = 'unsupported_token_type'