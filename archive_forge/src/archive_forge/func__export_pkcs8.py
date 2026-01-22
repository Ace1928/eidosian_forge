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
def _export_pkcs8(self, **kwargs):
    from Cryptodome.IO import PKCS8
    if kwargs.get('passphrase', None) is not None and 'protection' not in kwargs:
        raise ValueError("At least the 'protection' parameter must be present")
    if self._is_eddsa():
        oid = self._curve.oid
        private_key = DerOctetString(self._seed).encode()
        params = None
    else:
        oid = '1.2.840.10045.2.1'
        private_key = self._export_rfc5915_private_der(include_ec_params=False)
        params = DerObjectId(self._curve.oid)
    result = PKCS8.wrap(private_key, oid, key_params=params, **kwargs)
    return result