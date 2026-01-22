from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class MissingCodeChallengeError(InvalidRequestError):
    """
    If the server requires Proof Key for Code Exchange (PKCE) by OAuth
    public clients and the client does not send the "code_challenge" in
    the request, the authorization endpoint MUST return the authorization
    error response with the "error" value set to "invalid_request".  The
    "error_description" or the response of "error_uri" SHOULD explain the
    nature of error, e.g., code challenge required.
    """
    description = 'Code challenge required.'