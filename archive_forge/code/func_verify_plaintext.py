from __future__ import absolute_import, unicode_literals
import binascii
import hashlib
import hmac
import logging
from oauthlib.common import (extract_params, safe_string_equals, unicode_type,
from . import utils
def verify_plaintext(request, client_secret=None, resource_owner_secret=None):
    """Verify a PLAINTEXT signature.

    Per `section 3.4`_ of the spec.

    .. _`section 3.4`: https://tools.ietf.org/html/rfc5849#section-3.4
    """
    signature = sign_plaintext(client_secret, resource_owner_secret)
    match = safe_string_equals(signature, request.signature)
    if not match:
        log.debug('Verify PLAINTEXT failed')
    return match