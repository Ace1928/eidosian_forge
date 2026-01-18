from __future__ import absolute_import, unicode_literals
import logging
def validate_id_token(self, token, scopes, request):
    """Ensure the id token is valid and authorized access to scopes.

        OpenID connect core 1.0 describe how to validate an id_token:
            -
            http://openid.net/specs/openid-connect-core-1_0.html#IDTokenValidation
            -
            http://openid.net/specs/openid-connect-core-1_0.html#ImplicitIDTValidation
            -
            http://openid.net/specs/openid-connect-core-1_0.html#HybridIDTValidation
            -
            http://openid.net/specs/openid-connect-core-1_0.html#HybridIDTValidation2

        :param token: Unicode Bearer token.
        :param scopes: List of scopes (defined by you).
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :rtype: True or False

        Method is indirectly used by all core OpenID connect JWT token issuing
        grant types:
            - Authorization Code Grant
            - Implicit Grant
            - Hybrid Grant
        """
    raise NotImplementedError('Subclasses must implement this method.')