from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class LoginRequired(OAuth2Error):
    """
    The Authorization Server requires End-User authentication.

    This error MAY be returned when the prompt parameter value in the
    Authentication Request is none, but the Authentication Request cannot be
    completed without displaying a user interface for End-User authentication.
    """
    error = 'login_required'