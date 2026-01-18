from __future__ import absolute_import, unicode_literals
import copy
import json
import logging
from ....common import unicode_type
from .base import BaseEndpoint, catch_errors_and_unavailability
from .authorization import AuthorizationEndpoint
from .introspect import IntrospectEndpoint
from .token import TokenEndpoint
from .revocation import RevocationEndpoint
from .. import grant_types
def validate_metadata_token(self, claims, endpoint):
    """
        If the token endpoint is used in the grant type, the value of this
        parameter MUST be the same as the value of the "grant_type"
        parameter passed to the token endpoint defined in the grant type
        definition.
        """
    self._grant_types.extend(endpoint._grant_types.keys())
    claims.setdefault('token_endpoint_auth_methods_supported', ['client_secret_post', 'client_secret_basic'])
    self.validate_metadata(claims, 'token_endpoint_auth_methods_supported', is_list=True)
    self.validate_metadata(claims, 'token_endpoint_auth_signing_alg_values_supported', is_list=True)
    self.validate_metadata(claims, 'token_endpoint', is_required=True, is_url=True)