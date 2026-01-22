from __future__ import absolute_import, unicode_literals
import logging
from oauthlib.oauth2.rfc6749.grant_types.authorization_code import AuthorizationCodeGrant as OAuth2AuthorizationCodeGrant
from oauthlib.oauth2.rfc6749.errors import InvalidRequestError
from .base import GrantTypeBase
from ..request_validator import RequestValidator
class HybridGrant(GrantTypeBase):

    def __init__(self, request_validator=None, **kwargs):
        self.request_validator = request_validator or RequestValidator()
        self.proxy_target = OAuth2AuthorizationCodeGrant(request_validator=request_validator, **kwargs)
        self.proxy_target.default_response_mode = 'fragment'
        self.register_response_type('code id_token')
        self.register_response_type('code token')
        self.register_response_type('code id_token token')
        self.custom_validators.post_auth.append(self.openid_authorization_validator)
        self.register_code_modifier(self.add_token)
        self.register_code_modifier(self.add_id_token)
        self.register_token_modifier(self.add_id_token)

    def openid_authorization_validator(self, request):
        """Additional validation when following the Authorization Code flow."""
        request_info = super(HybridGrant, self).openid_authorization_validator(request)
        if not request_info:
            return request_info
        if request.response_type in ['code id_token', 'code id_token token']:
            if not request.nonce:
                raise InvalidRequestError(request=request, description='Request is missing mandatory nonce parameter.')
        return request_info