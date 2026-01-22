import binascii
import hashlib
import hmac
import ipaddress
import logging
import urllib.parse as urlparse
import warnings
from oauthlib.common import extract_params, safe_string_equals, urldecode
from . import utils
def _verify_rsa(hash_algorithm_name: str, request, rsa_public_key: str):
    """
    Verify a base64 encoded signature for a RSA-based signature method.

    The ``alg`` is used to calculate the digest over the signature base string.
    For the "RSA_SHA1" signature method, the alg must be SHA-1. While OAuth 1.0a
    only defines the RSA-SHA1 signature method, this function can be used for
    other non-standard signature methods that only differ from RSA-SHA1 by the
    digest algorithm.

    Verification for the RSA-SHA1 signature method is defined in
    `section 3.4.3`_ of RFC 5849.

    .. _`section 3.4.3`: https://tools.ietf.org/html/rfc5849#section-3.4.3

        To satisfy `RFC2616 section 5.2`_ item 1, the request argument's uri
        attribute MUST be an absolute URI whose netloc part identifies the
        origin server or gateway on which the resource resides. Any Host
        item of the request argument's headers dict attribute will be
        ignored.

        .. _`RFC2616 Sec 5.2`: https://tools.ietf.org/html/rfc2616#section-5.2
    """
    try:
        norm_params = normalize_parameters(request.params)
        bs_uri = base_string_uri(request.uri)
        sig_base_str = signature_base_string(request.http_method, bs_uri, norm_params)
        sig = binascii.a2b_base64(request.signature.encode('ascii'))
        alg = _get_jwt_rsa_algorithm(hash_algorithm_name)
        key = _prepare_key_plus(alg, rsa_public_key)
        verify_ok = alg.verify(sig_base_str.encode('ascii'), key, sig)
        if not verify_ok:
            log.debug('Verify failed: RSA with ' + alg.hash_alg.name + ': signature base string=%s' + sig_base_str)
        return verify_ok
    except UnicodeError:
        return False