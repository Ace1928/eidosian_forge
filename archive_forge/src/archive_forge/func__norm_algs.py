import logging; log = logging.getLogger(__name__)
from passlib.utils import consteq, saslprep, to_native_str, splitcomma
from passlib.utils.binary import ab64_decode, ab64_encode
from passlib.utils.compat import bascii_to_str, iteritems, u, native_string_types
from passlib.crypto.digest import pbkdf2_hmac, norm_hash_name
import passlib.utils.handlers as uh
@classmethod
def _norm_algs(cls, algs):
    """normalize algs parameter"""
    if isinstance(algs, native_string_types):
        algs = splitcomma(algs)
    algs = sorted((norm_hash_name(alg, 'iana') for alg in algs))
    if any((len(alg) > 9 for alg in algs)):
        raise ValueError('SCRAM limits alg names to max of 9 characters')
    if 'sha-1' not in algs:
        raise ValueError('sha-1 must be in algorithm list of scram hash')
    return algs