from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class FatalClientError(OAuth2Error):
    """
    Errors during authorization where user should not be redirected back.

    If the request fails due to a missing, invalid, or mismatching
    redirection URI, or if the client identifier is missing or invalid,
    the authorization server SHOULD inform the resource owner of the
    error and MUST NOT automatically redirect the user-agent to the
    invalid redirection URI.

    Instead the user should be informed of the error by the provider itself.
    """
    pass