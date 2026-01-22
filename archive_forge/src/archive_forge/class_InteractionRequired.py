from __future__ import unicode_literals
from oauthlib.oauth2.rfc6749.errors import FatalClientError, OAuth2Error
class InteractionRequired(OpenIDClientError):
    """
    The Authorization Server requires End-User interaction to proceed.

    This error MAY be returned when the prompt parameter value in the
    Authentication Request is none, but the Authentication Request cannot be
    completed without displaying a user interface for End-User interaction.
    """
    error = 'interaction_required'
    status_code = 401