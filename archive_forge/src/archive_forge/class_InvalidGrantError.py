from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class InvalidGrantError(OAuth2Error):
    """
    The provided authorization grant (e.g. authorization code, resource
    owner credentials) or refresh token is invalid, expired, revoked, does
    not match the redirection URI used in the authorization request, or was
    issued to another client.

    https://tools.ietf.org/html/rfc6749#section-5.2
    """
    error = 'invalid_grant'
    status_code = 400