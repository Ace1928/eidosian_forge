from __future__ import absolute_import, unicode_literals
from ..grant_types import (AuthorizationCodeGrant, ClientCredentialsGrant,
from ..tokens import BearerToken
from .authorization import AuthorizationEndpoint
from .introspect import IntrospectEndpoint
from .resource import ResourceEndpoint
from .revocation import RevocationEndpoint
from .token import TokenEndpoint
class WebApplicationServer(AuthorizationEndpoint, IntrospectEndpoint, TokenEndpoint, ResourceEndpoint, RevocationEndpoint):
    """An all-in-one endpoint featuring Authorization code grant and Bearer tokens."""

    def __init__(self, request_validator, token_generator=None, token_expires_in=None, refresh_token_generator=None, **kwargs):
        """Construct a new web application server.

        :param request_validator: An implementation of
                                  oauthlib.oauth2.RequestValidator.
        :param token_expires_in: An int or a function to generate a token
                                 expiration offset (in seconds) given a
                                 oauthlib.common.Request object.
        :param token_generator: A function to generate a token from a request.
        :param refresh_token_generator: A function to generate a token from a
                                        request for the refresh token.
        :param kwargs: Extra parameters to pass to authorization-,
                       token-, resource-, and revocation-endpoint constructors.
        """
        auth_grant = AuthorizationCodeGrant(request_validator)
        refresh_grant = RefreshTokenGrant(request_validator)
        bearer = BearerToken(request_validator, token_generator, token_expires_in, refresh_token_generator)
        AuthorizationEndpoint.__init__(self, default_response_type='code', response_types={'code': auth_grant}, default_token_type=bearer)
        TokenEndpoint.__init__(self, default_grant_type='authorization_code', grant_types={'authorization_code': auth_grant, 'refresh_token': refresh_grant}, default_token_type=bearer)
        ResourceEndpoint.__init__(self, default_token='Bearer', token_types={'Bearer': bearer})
        RevocationEndpoint.__init__(self, request_validator)
        IntrospectEndpoint.__init__(self, request_validator)