from __future__ import unicode_literals
from oauthlib.oauth2.rfc6749.errors import FatalClientError, OAuth2Error
class InvalidRequestObject(OpenIDClientError):
    """
    The request parameter contains an invalid Request Object.
    """
    error = 'invalid_request_object'
    description = 'The request parameter contains an invalid Request Object.'