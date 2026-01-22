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
def _import_openssh_private_ecc(data, password):
    from ._openssh import import_openssh_private_generic, read_bytes, read_string, check_padding
    key_type, decrypted = import_openssh_private_generic(data, password)
    eddsa_keys = {'ssh-ed25519': ('Ed25519', _import_ed25519_public_key, 32)}
    if key_type.startswith('ecdsa-sha2'):
        ecdsa_curve_name, decrypted = read_string(decrypted)
        if ecdsa_curve_name not in _curves:
            raise UnsupportedEccFeature('Unsupported ECC curve %s' % ecdsa_curve_name)
        curve = _curves[ecdsa_curve_name]
        modulus_bytes = (curve.modulus_bits + 7) // 8
        public_key, decrypted = read_bytes(decrypted)
        if bord(public_key[0]) != 4:
            raise ValueError('Only uncompressed OpenSSH EC keys are supported')
        if len(public_key) != 2 * modulus_bytes + 1:
            raise ValueError('Incorrect public key length')
        point_x = Integer.from_bytes(public_key[1:1 + modulus_bytes])
        point_y = Integer.from_bytes(public_key[1 + modulus_bytes:])
        private_key, decrypted = read_bytes(decrypted)
        d = Integer.from_bytes(private_key)
        params = {'d': d, 'curve': ecdsa_curve_name}
    elif key_type in eddsa_keys:
        curve_name, import_eddsa_public_key, seed_len = eddsa_keys[key_type]
        public_key, decrypted = read_bytes(decrypted)
        point_x, point_y = import_eddsa_public_key(public_key)
        private_public_key, decrypted = read_bytes(decrypted)
        seed = private_public_key[:seed_len]
        params = {'seed': seed, 'curve': curve_name}
    else:
        raise ValueError('Unsupport SSH agent key type:' + key_type)
    _, padded = read_string(decrypted)
    check_padding(padded)
    return construct(point_x=point_x, point_y=point_y, **params)