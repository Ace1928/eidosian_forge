import abc
import base64
import enum
import json
import six
from google.auth import exceptions
class ClientAuthentication(object):
    """Defines the client authentication credentials for basic and request-body
    types based on https://tools.ietf.org/html/rfc6749#section-2.3.1.
    """

    def __init__(self, client_auth_type, client_id, client_secret=None):
        """Instantiates a client authentication object containing the client ID
        and secret credentials for basic and response-body auth.

        Args:
            client_auth_type (google.oauth2.oauth_utils.ClientAuthType): The
                client authentication type.
            client_id (str): The client ID.
            client_secret (Optional[str]): The client secret.
        """
        self.client_auth_type = client_auth_type
        self.client_id = client_id
        self.client_secret = client_secret