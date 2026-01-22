from __future__ import absolute_import, unicode_literals
import time
from oauthlib.common import Request, generate_token
from .. import (CONTENT_TYPE_FORM_URLENCODED, SIGNATURE_HMAC, SIGNATURE_RSA,
def _check_transport_security(self, request):
    if self.request_validator.enforce_ssl and (not request.uri.lower().startswith('https://')):
        raise errors.InsecureTransportError()