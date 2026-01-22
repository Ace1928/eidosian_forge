import hashlib
import os
from rsa._compat import range
from rsa import common, transform, core
def _find_method_hash(clearsig):
    """Finds the hash method.

    :param clearsig: full padded ASN1 and hash.
    :return: the used hash method.
    :raise VerificationFailed: when the hash method cannot be found
    """
    for hashname, asn1code in HASH_ASN1.items():
        if asn1code in clearsig:
            return hashname
    raise VerificationError('Verification failed')