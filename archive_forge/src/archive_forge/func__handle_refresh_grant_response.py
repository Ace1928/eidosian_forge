import datetime
import json
import six
from six.moves import http_client
from six.moves import urllib
from google.auth import _exponential_backoff
from google.auth import _helpers
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport
def _handle_refresh_grant_response(response_data, refresh_token):
    """Extract tokens from refresh grant response.

    Args:
        response_data (Mapping[str, str]): Refresh grant response data.
        refresh_token (str): Current refresh token.

    Returns:
        Tuple[str, str, Optional[datetime], Mapping[str, str]]: The access token,
            refresh token, expiration, and additional data returned by the token
            endpoint. If response_data doesn't have refresh token, then the current
            refresh token will be returned.

    Raises:
        google.auth.exceptions.RefreshError: If the token endpoint returned
            an error.
    """
    try:
        access_token = response_data['access_token']
    except KeyError as caught_exc:
        new_exc = exceptions.RefreshError('No access token in response.', response_data, retryable=False)
        six.raise_from(new_exc, caught_exc)
    refresh_token = response_data.get('refresh_token', refresh_token)
    expiry = _parse_expiry(response_data)
    return (access_token, refresh_token, expiry, response_data)