import json
from six.moves import http_client
from six.moves import urllib
from google.oauth2 import utils
Exchanges a refresh token for an access token based on the
        RFC6749 spec.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            subject_token (str): The OAuth 2.0 refresh token.
        