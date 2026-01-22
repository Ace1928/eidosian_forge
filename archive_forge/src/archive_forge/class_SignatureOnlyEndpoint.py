from __future__ import absolute_import, unicode_literals
import logging
from .. import errors
from .base import BaseEndpoint
class SignatureOnlyEndpoint(BaseEndpoint):
    """An endpoint only responsible for verifying an oauth signature."""

    def validate_request(self, uri, http_method='GET', body=None, headers=None):
        """Validate a signed OAuth request.

        :param uri: The full URI of the token request.
        :param http_method: A valid HTTP verb, i.e. GET, POST, PUT, HEAD, etc.
        :param body: The request body as a string.
        :param headers: The request headers as a dict.
        :returns: A tuple of 2 elements.
                  1. True if valid, False otherwise.
                  2. An oauthlib.common.Request object.
        """
        try:
            request = self._create_request(uri, http_method, body, headers)
        except errors.OAuth1Error as err:
            log.info('Exception caught while validating request, %s.' % err)
            return (False, None)
        try:
            self._check_transport_security(request)
            self._check_mandatory_parameters(request)
        except errors.OAuth1Error as err:
            log.info('Exception caught while validating request, %s.' % err)
            return (False, request)
        if not self.request_validator.validate_timestamp_and_nonce(request.client_key, request.timestamp, request.nonce, request):
            log.debug('[Failure] verification failed: timestamp/nonce')
            return (False, request)
        valid_client = self.request_validator.validate_client_key(request.client_key, request)
        if not valid_client:
            request.client_key = self.request_validator.dummy_client
        valid_signature = self._check_signature(request)
        request.validator_log['client'] = valid_client
        request.validator_log['signature'] = valid_signature
        v = all((valid_client, valid_signature))
        if not v:
            log.info('[Failure] request verification failed.')
            log.info('Valid client: %s', valid_client)
            log.info('Valid signature: %s', valid_signature)
        return (v, request)