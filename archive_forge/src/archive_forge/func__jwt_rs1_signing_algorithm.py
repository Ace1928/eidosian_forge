from __future__ import absolute_import, unicode_literals
import binascii
import hashlib
import hmac
import logging
from oauthlib.common import (extract_params, safe_string_equals, unicode_type,
from . import utils
def _jwt_rs1_signing_algorithm():
    global _jwtrs1
    if _jwtrs1 is None:
        import jwt.algorithms as jwtalgo
        _jwtrs1 = jwtalgo.RSAAlgorithm(jwtalgo.hashes.SHA1)
    return _jwtrs1