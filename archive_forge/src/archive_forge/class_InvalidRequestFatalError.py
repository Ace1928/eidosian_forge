from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class InvalidRequestFatalError(FatalClientError):
    """
    For fatal errors, the request is missing a required parameter, includes
    an invalid parameter value, includes a parameter more than once, or is
    otherwise malformed.
    """
    error = 'invalid_request'