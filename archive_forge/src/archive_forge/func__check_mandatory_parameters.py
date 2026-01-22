from __future__ import absolute_import, unicode_literals
import time
from oauthlib.common import Request, generate_token
from .. import (CONTENT_TYPE_FORM_URLENCODED, SIGNATURE_HMAC, SIGNATURE_RSA,
def _check_mandatory_parameters(self, request):
    if not all((request.signature, request.client_key, request.nonce, request.timestamp, request.signature_method)):
        raise errors.InvalidRequestError(description='Missing mandatory OAuth parameters.')
    if not request.signature_method in self.request_validator.allowed_signature_methods:
        raise errors.InvalidSignatureMethodError(description='Invalid signature, %s not in %r.' % (request.signature_method, self.request_validator.allowed_signature_methods))
    if 'oauth_version' in request.oauth_params and request.oauth_params['oauth_version'] != '1.0':
        raise errors.InvalidRequestError(description='Invalid OAuth version.')
    if len(request.timestamp) != 10:
        raise errors.InvalidRequestError(description='Invalid timestamp size')
    try:
        ts = int(request.timestamp)
    except ValueError:
        raise errors.InvalidRequestError(description='Timestamp must be an integer.')
    else:
        if abs(time.time() - ts) > self.request_validator.timestamp_lifetime:
            raise errors.InvalidRequestError(description='Timestamp given is invalid, differ from allowed by over %s seconds.' % self.request_validator.timestamp_lifetime)
    if not self.request_validator.check_client_key(request.client_key):
        raise errors.InvalidRequestError(description='Invalid client key format.')
    if not self.request_validator.check_nonce(request.nonce):
        raise errors.InvalidRequestError(description='Invalid nonce format.')