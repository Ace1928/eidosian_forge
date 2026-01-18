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
def validate_metadata_server(self):
    """
        Authorization servers can have metadata describing their
        configuration.  The following authorization server metadata values
        are used by this specification. More details can be found in
        `RFC8414 section 2`_ :

       issuer
          REQUIRED

       authorization_endpoint
          URL of the authorization server's authorization endpoint
          [`RFC6749#Authorization`_].  This is REQUIRED unless no grant types
          are supported
          that use the authorization endpoint.

       token_endpoint
          URL of the authorization server's token endpoint [`RFC6749#Token`_].
          This
          is REQUIRED unless only the implicit grant type is supported.

       scopes_supported
          RECOMMENDED.

       response_types_supported
          REQUIRED.

       * Other OPTIONAL fields:
       jwks_uri
       registration_endpoint
       response_modes_supported

       grant_types_supported
          OPTIONAL.  JSON array containing a list of the OAuth 2.0 grant
          type values that this authorization server supports.  The array
          values used are the same as those used with the "grant_types"
          parameter defined by "OAuth 2.0 Dynamic Client Registration
          Protocol" [`RFC7591`_].  If omitted, the default value is
          "["authorization_code", "implicit"]".

       token_endpoint_auth_methods_supported

       token_endpoint_auth_signing_alg_values_supported

       service_documentation

       ui_locales_supported

       op_policy_uri

       op_tos_uri

       revocation_endpoint

       revocation_endpoint_auth_methods_supported

       revocation_endpoint_auth_signing_alg_values_supported

       introspection_endpoint

       introspection_endpoint_auth_methods_supported

       introspection_endpoint_auth_signing_alg_values_supported

       code_challenge_methods_supported

       Additional authorization server metadata parameters MAY also be used.
       Some are defined by other specifications, such as OpenID Connect
       Discovery 1.0 [`OpenID.Discovery`_].

        .. _`RFC8414 section 2`: https://tools.ietf.org/html/rfc8414#section-2
        .. _`RFC6749#Authorization`:
        https://tools.ietf.org/html/rfc6749#section-3.1
        .. _`RFC6749#Token`: https://tools.ietf.org/html/rfc6749#section-3.2
        .. _`RFC7591`: https://tools.ietf.org/html/rfc7591
        .. _`OpenID.Discovery`:
        https://openid.net/specs/openid-connect-discovery-1_0.html
        """
    claims = copy.deepcopy(self.initial_claims)
    self.validate_metadata(claims, 'issuer', is_required=True, is_issuer=True)
    self.validate_metadata(claims, 'jwks_uri', is_url=True)
    self.validate_metadata(claims, 'scopes_supported', is_list=True)
    self.validate_metadata(claims, 'service_documentation', is_url=True)
    self.validate_metadata(claims, 'ui_locales_supported', is_list=True)
    self.validate_metadata(claims, 'op_policy_uri', is_url=True)
    self.validate_metadata(claims, 'op_tos_uri', is_url=True)
    self._grant_types = []
    for endpoint in self.endpoints:
        if isinstance(endpoint, TokenEndpoint):
            self.validate_metadata_token(claims, endpoint)
        if isinstance(endpoint, AuthorizationEndpoint):
            self.validate_metadata_authorization(claims, endpoint)
        if isinstance(endpoint, RevocationEndpoint):
            self.validate_metadata_revocation(claims, endpoint)
        if isinstance(endpoint, IntrospectEndpoint):
            self.validate_metadata_introspection(claims, endpoint)
    claims.setdefault('grant_types_supported', self._grant_types)
    self.validate_metadata(claims, 'grant_types_supported', is_list=True)
    return claims