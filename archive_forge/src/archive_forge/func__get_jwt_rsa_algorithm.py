import binascii
import hashlib
import hmac
import ipaddress
import logging
import urllib.parse as urlparse
import warnings
from oauthlib.common import extract_params, safe_string_equals, urldecode
from . import utils
def _get_jwt_rsa_algorithm(hash_algorithm_name: str):
    """
    Obtains an RSAAlgorithm object that implements RSA with the hash algorithm.

    This method maintains the ``_jwt_rsa`` cache.

    Returns a jwt.algorithm.RSAAlgorithm.
    """
    if hash_algorithm_name in _jwt_rsa:
        return _jwt_rsa[hash_algorithm_name]
    else:
        import jwt.algorithms as jwt_algorithms
        m = {'SHA-1': jwt_algorithms.hashes.SHA1, 'SHA-256': jwt_algorithms.hashes.SHA256, 'SHA-512': jwt_algorithms.hashes.SHA512}
        v = jwt_algorithms.RSAAlgorithm(m[hash_algorithm_name])
        _jwt_rsa[hash_algorithm_name] = v
        return v