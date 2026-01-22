from __future__ import print_function
import re
import struct
import binascii
from collections import namedtuple
from Cryptodome.Util.py3compat import bord, tobytes, tostr, bchr, is_string
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Util.asn1 import (DerObjectId, DerOctetString, DerSequence,
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.Random import get_random_bytes
from Cryptodome.Random.random import getrandbits
def _import_pkcs8(encoded, passphrase):
    from Cryptodome.IO import PKCS8
    algo_oid, private_key, params = PKCS8.unwrap(encoded, passphrase)
    nist_p_oids = ('1.2.840.10045.2.1', '1.3.132.1.12', '1.3.132.1.13')
    eddsa_oids = {'1.3.101.112': 'Ed25519', '1.3.101.113': 'Ed448'}
    if algo_oid in nist_p_oids:
        curve_oid = DerObjectId().decode(params).value
        return _import_rfc5915_der(private_key, passphrase, curve_oid)
    elif algo_oid in eddsa_oids:
        if params is not None:
            raise ValueError('EdDSA ECC private key must not have parameters')
        curve_oid = None
        seed = DerOctetString().decode(private_key).payload
        return construct(curve=eddsa_oids[algo_oid], seed=seed)
    else:
        raise UnsupportedEccFeature('Unsupported ECC purpose (OID: %s)' % algo_oid)