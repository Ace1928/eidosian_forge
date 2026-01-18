from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import sys
from google_reauth import challenges
from google_reauth import errors
from google_reauth import _helpers
from google_reauth import _reauth_client
from six.moves import http_client
from six.moves import range
Refresh the access_token using the refresh_token.

    Args:
        http_request: callable to run http requests. Accepts uri, method, body
            and headers. Returns a tuple: (response, content)
        client_id: client id to get access token for reauth scope.
        client_secret: client secret for the client_id
        refresh_token: refresh token to refresh access token
        token_uri: uri to refresh access token
        scopes: scopes required by the client application

    Returns:
        Tuple[str, str, str, Optional[str], Optional[str], Optional[str]]: The
            rapt token, the access token, new refresh token, expiration,
            token id and response content returned by the token endpoint.
    Raises:
        errors.ReauthError if reauth failed
        errors.HttpAccessTokenRefreshError it access token refresh failed
    