from __future__ import absolute_import, unicode_literals
import time
import warnings
from oauthlib.common import generate_token
from oauthlib.oauth2.rfc6749 import tokens
from oauthlib.oauth2.rfc6749.errors import (InsecureTransportError,
from oauthlib.oauth2.rfc6749.parameters import (parse_token_response,
from oauthlib.oauth2.rfc6749.utils import is_secure_transport
@property
def token_types(self):
    """Supported token types and their respective methods

        Additional tokens can be supported by extending this dictionary.

        The Bearer token spec is stable and safe to use.

        The MAC token spec is not yet stable and support for MAC tokens
        is experimental and currently matching version 00 of the spec.
        """
    return {'Bearer': self._add_bearer_token, 'MAC': self._add_mac_token}