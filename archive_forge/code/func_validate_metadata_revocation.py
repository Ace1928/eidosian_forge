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
def validate_metadata_revocation(self, claims, endpoint):
    claims.setdefault('revocation_endpoint_auth_methods_supported', ['client_secret_post', 'client_secret_basic'])
    self.validate_metadata(claims, 'revocation_endpoint_auth_methods_supported', is_list=True)
    self.validate_metadata(claims, 'revocation_endpoint_auth_signing_alg_values_supported', is_list=True)
    self.validate_metadata(claims, 'revocation_endpoint', is_required=True, is_url=True)