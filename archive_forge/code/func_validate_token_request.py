from __future__ import absolute_import, unicode_literals
import json
import logging
from .. import errors
from ..request_validator import RequestValidator
from .base import GrantTypeBase
def validate_token_request(self, request):
    """
        :param request: OAuthlib request.
        :type request: oauthlib.common.Request

        The client makes a request to the token endpoint by adding the
        following parameters using the "application/x-www-form-urlencoded"
        format per Appendix B with a character encoding of UTF-8 in the HTTP
        request entity-body:

        grant_type
                REQUIRED.  Value MUST be set to "password".

        username
                REQUIRED.  The resource owner username.

        password
                REQUIRED.  The resource owner password.

        scope
                OPTIONAL.  The scope of the access request as described by
                `Section 3.3`_.

        If the client type is confidential or the client was issued client
        credentials (or assigned other authentication requirements), the
        client MUST authenticate with the authorization server as described
        in `Section 3.2.1`_.

        The authorization server MUST:

        o  require client authentication for confidential clients or for any
            client that was issued client credentials (or with other
            authentication requirements),

        o  authenticate the client if client authentication is included, and

        o  validate the resource owner password credentials using its
            existing password validation algorithm.

        Since this access token request utilizes the resource owner's
        password, the authorization server MUST protect the endpoint against
        brute force attacks (e.g., using rate-limitation or generating
        alerts).

        .. _`Section 3.3`: https://tools.ietf.org/html/rfc6749#section-3.3
        .. _`Section 3.2.1`: https://tools.ietf.org/html/rfc6749#section-3.2.1
        """
    for validator in self.custom_validators.pre_token:
        validator(request)
    for param in ('grant_type', 'username', 'password'):
        if not getattr(request, param, None):
            raise errors.InvalidRequestError('Request is missing %s parameter.' % param, request=request)
    for param in ('grant_type', 'username', 'password', 'scope'):
        if param in request.duplicate_params:
            raise errors.InvalidRequestError(description='Duplicate %s parameter.' % param, request=request)
    if not request.grant_type == 'password':
        raise errors.UnsupportedGrantTypeError(request=request)
    log.debug('Validating username %s.', request.username)
    if not self.request_validator.validate_user(request.username, request.password, request.client, request):
        raise errors.InvalidGrantError('Invalid credentials given.', request=request)
    elif not hasattr(request.client, 'client_id'):
        raise NotImplementedError('Validate user must set the request.client.client_id attribute in authenticate_client.')
    log.debug('Authorizing access to user %r.', request.user)
    self.validate_grant_type(request)
    if request.client:
        request.client_id = request.client_id or request.client.client_id
    self.validate_scopes(request)
    for validator in self.custom_validators.post_token:
        validator(request)