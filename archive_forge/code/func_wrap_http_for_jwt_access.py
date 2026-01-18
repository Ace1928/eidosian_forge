import logging
import httplib2
import six
from six.moves import http_client
from oauth2client import _helpers
def wrap_http_for_jwt_access(credentials, http):
    """Prepares an HTTP object's request method for JWT access.

    Wraps HTTP requests with logic to catch auth failures (typically
    identified via a 401 status code). In the event of failure, tries
    to refresh the token used and then retry the original request.

    Args:
        credentials: _JWTAccessCredentials, the credentials used to identify
                     a service account that uses JWT access tokens.
        http: httplib2.Http, an http object to be used to make
              auth requests.
    """
    orig_request_method = http.request
    wrap_http_for_auth(credentials, http)
    authenticated_request_method = http.request

    def new_request(uri, method='GET', body=None, headers=None, redirections=httplib2.DEFAULT_MAX_REDIRECTS, connection_type=None):
        if 'aud' in credentials._kwargs:
            if credentials.access_token is None or credentials.access_token_expired:
                credentials.refresh(None)
            return request(authenticated_request_method, uri, method, body, headers, redirections, connection_type)
        else:
            headers = _initialize_headers(headers)
            _apply_user_agent(headers, credentials.user_agent)
            uri_root = uri.split('?', 1)[0]
            token, unused_expiry = credentials._create_token({'aud': uri_root})
            headers['Authorization'] = 'Bearer ' + token
            return request(orig_request_method, uri, method, body, clean_headers(headers), redirections, connection_type)
    http.request = new_request
    http.request.credentials = credentials