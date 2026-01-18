from __future__ import absolute_import, unicode_literals
import logging
from oauthlib.common import urlencode
from .. import errors
from .base import BaseEndpoint
def validate_access_token_request(self, request):
    """Validate an access token request.

        :param request: OAuthlib request.
        :type request: oauthlib.common.Request
        :raises: OAuth1Error if the request is invalid.
        :returns: A tuple of 2 elements.
                  1. The validation result (True or False).
                  2. The request object.
        """
    self._check_transport_security(request)
    self._check_mandatory_parameters(request)
    if not request.resource_owner_key:
        raise errors.InvalidRequestError(description='Missing resource owner.')
    if not self.request_validator.check_request_token(request.resource_owner_key):
        raise errors.InvalidRequestError(description='Invalid resource owner key format.')
    if not request.verifier:
        raise errors.InvalidRequestError(description='Missing verifier.')
    if not self.request_validator.check_verifier(request.verifier):
        raise errors.InvalidRequestError(description='Invalid verifier format.')
    if not self.request_validator.validate_timestamp_and_nonce(request.client_key, request.timestamp, request.nonce, request, request_token=request.resource_owner_key):
        return (False, request)
    valid_client = self.request_validator.validate_client_key(request.client_key, request)
    if not valid_client:
        request.client_key = self.request_validator.dummy_client
    valid_resource_owner = self.request_validator.validate_request_token(request.client_key, request.resource_owner_key, request)
    if not valid_resource_owner:
        request.resource_owner_key = self.request_validator.dummy_request_token
    valid_verifier = self.request_validator.validate_verifier(request.client_key, request.resource_owner_key, request.verifier, request)
    valid_signature = self._check_signature(request, is_token_request=True)
    request.validator_log['client'] = valid_client
    request.validator_log['resource_owner'] = valid_resource_owner
    request.validator_log['verifier'] = valid_verifier
    request.validator_log['signature'] = valid_signature
    v = all((valid_client, valid_resource_owner, valid_verifier, valid_signature))
    if not v:
        log.info('[Failure] request verification failed.')
        log.info('Valid client:, %s', valid_client)
        log.info('Valid token:, %s', valid_resource_owner)
        log.info('Valid verifier:, %s', valid_verifier)
        log.info('Valid signature:, %s', valid_signature)
    return (v, request)