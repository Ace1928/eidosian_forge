from __future__ import absolute_import, unicode_literals
import base64
import hashlib
import json
import logging
from oauthlib import common
from .. import errors
from .base import GrantTypeBase
def validate_authorization_request(self, request):
    """Check the authorization request for normal and fatal errors.

        A normal error could be a missing response_type parameter or the client
        attempting to access scope it is not allowed to ask authorization for.
        Normal errors can safely be included in the redirection URI and
        sent back to the client.

        Fatal errors occur when the client_id or redirect_uri is invalid or
        missing. These must be caught by the provider and handled, how this
        is done is outside of the scope of OAuthLib but showing an error
        page describing the issue is a good idea.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        """
    for param in ('client_id', 'response_type', 'redirect_uri', 'scope', 'state'):
        try:
            duplicate_params = request.duplicate_params
        except ValueError:
            raise errors.InvalidRequestFatalError(description='Unable to parse query string', request=request)
        if param in duplicate_params:
            raise errors.InvalidRequestFatalError(description='Duplicate %s parameter.' % param, request=request)
    if not request.client_id:
        raise errors.MissingClientIdError(request=request)
    if not self.request_validator.validate_client_id(request.client_id, request):
        raise errors.InvalidClientIdError(request=request)
    log.debug('Validating redirection uri %s for client %s.', request.redirect_uri, request.client_id)
    self._handle_redirects(request)
    request_info = {}
    for validator in self.custom_validators.pre_auth:
        request_info.update(validator(request))
    if request.response_type is None:
        raise errors.MissingResponseTypeError(request=request)
    elif not 'code' in request.response_type and request.response_type != 'none':
        raise errors.UnsupportedResponseTypeError(request=request)
    if not self.request_validator.validate_response_type(request.client_id, request.response_type, request.client, request):
        log.debug('Client %s is not authorized to use response_type %s.', request.client_id, request.response_type)
        raise errors.UnauthorizedClientError(request=request)
    if self.request_validator.is_pkce_required(request.client_id, request) is True:
        if request.code_challenge is None:
            raise errors.MissingCodeChallengeError(request=request)
    if request.code_challenge is not None:
        if request.code_challenge_method is None:
            request.code_challenge_method = 'plain'
        if request.code_challenge_method not in self._code_challenge_methods:
            raise errors.UnsupportedCodeChallengeMethodError(request=request)
    self.validate_scopes(request)
    request_info.update({'client_id': request.client_id, 'redirect_uri': request.redirect_uri, 'response_type': request.response_type, 'state': request.state, 'request': request})
    for validator in self.custom_validators.post_auth:
        request_info.update(validator(request))
    return (request.scopes, request_info)