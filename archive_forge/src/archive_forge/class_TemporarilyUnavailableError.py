from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class TemporarilyUnavailableError(OAuth2Error):
    """
    The authorization server is currently unable to handle the request
    due to a temporary overloading or maintenance of the server.
    (This error code is needed because a 503 Service Unavailable HTTP
    status code cannot be returned to the client via a HTTP redirect.)
    """
    error = 'temporarily_unavailable'