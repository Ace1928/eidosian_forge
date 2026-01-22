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
def _token_endpoint_request(request, token_uri, body, access_token=None, use_json=False, can_retry=True, **kwargs):
    """Makes a request to the OAuth 2.0 authorization server's token endpoint.

    Args:
        request (google.auth.transport.Request): A callable used to make
            HTTP requests.
        token_uri (str): The OAuth 2.0 authorizations server's token endpoint
            URI.
        body (Mapping[str, str]): The parameters to send in the request body.
        access_token (Optional(str)): The access token needed to make the request.
        use_json (Optional(bool)): Use urlencoded format or json format for the
            content type. The default value is False.
        can_retry (bool): Enable or disable request retry behavior.
        kwargs: Additional arguments passed on to the request method. The
            kwargs will be passed to `requests.request` method, see:
            https://docs.python-requests.org/en/latest/api/#requests.request.
            For example, you can use `cert=("cert_pem_path", "key_pem_path")`
            to set up client side SSL certificate, and use
            `verify="ca_bundle_path"` to set up the CA certificates for sever
            side SSL certificate verification.

    Returns:
        Mapping[str, str]: The JSON-decoded response data.

    Raises:
        google.auth.exceptions.RefreshError: If the token endpoint returned
            an error.
    """
    response_status_ok, response_data, retryable_error = _token_endpoint_request_no_throw(request, token_uri, body, access_token=access_token, use_json=use_json, can_retry=can_retry, **kwargs)
    if not response_status_ok:
        _handle_error_response(response_data, retryable_error)
    return response_data