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
def _export_subjectPublicKeyInfo(self, compress):
    if self._is_eddsa():
        oid = self._curve.oid
        public_key = self._export_eddsa()
        params = None
    else:
        oid = '1.2.840.10045.2.1'
        public_key = self._export_SEC1(compress)
        params = DerObjectId(self._curve.oid)
    return _create_subject_public_key_info(oid, public_key, params)