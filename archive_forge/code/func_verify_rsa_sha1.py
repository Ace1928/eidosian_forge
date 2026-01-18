from __future__ import absolute_import, unicode_literals
import binascii
import hashlib
import hmac
import logging
from oauthlib.common import (extract_params, safe_string_equals, unicode_type,
from . import utils
def verify_rsa_sha1(request, rsa_public_key):
    """Verify a RSASSA-PKCS #1 v1.5 base64 encoded signature.

    Per `section 3.4.3`_ of the spec.

    Note this method requires the jwt and cryptography libraries.

    .. _`section 3.4.3`: https://tools.ietf.org/html/rfc5849#section-3.4.3

    To satisfy `RFC2616 section 5.2`_ item 1, the request argument's uri
    attribute MUST be an absolute URI whose netloc part identifies the
    origin server or gateway on which the resource resides. Any Host
    item of the request argument's headers dict attribute will be
    ignored.

    .. _`RFC2616 section 5.2`: https://tools.ietf.org/html/rfc2616#section-5.2
    """
    norm_params = normalize_parameters(request.params)
    uri = normalize_base_string_uri(request.uri)
    message = construct_base_string(request.http_method, uri, norm_params).encode('utf-8')
    sig = binascii.a2b_base64(request.signature.encode('utf-8'))
    alg = _jwt_rs1_signing_algorithm()
    key = _prepare_key_plus(alg, rsa_public_key)
    verify_ok = alg.verify(message, key, sig)
    if not verify_ok:
        log.debug('Verify RSA-SHA1 failed: sig base string: %s', message)
    return verify_ok