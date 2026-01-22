import json
from six.moves import urllib
from google_reauth import errors
Implements the OAuth 2.0 Refresh Grant with the addition of the reauth
    token.

    Args:
        http_request (Callable): callable to run http requests. Accepts uri,
            method, body and headers. Returns a tuple: (response, content)
        client_id (str): client id to get access token for reauth scope.
        client_secret (str): client secret for the client_id
        refresh_token (str): refresh token to refresh access token
        token_uri (str): uri to refresh access token
        scopes (str): scopes required by the client application as a
            comma-joined list.
        rapt (str): RAPT token
        headers (dict): headers for http request

    Returns:
        Tuple[str, dict]: http response and parsed response content.
    