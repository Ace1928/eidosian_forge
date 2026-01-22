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
def _import_public_der(ec_point, curve_oid=None, curve_name=None):
    """Convert an encoded EC point into an EccKey object

    ec_point: byte string with the EC point (SEC1-encoded)
    curve_oid: string with the name the curve
    curve_name: string with the OID of the curve

    Either curve_id or curve_name must be specified

    """
    for _curve_name, curve in _curves.items():
        if curve_oid and curve.oid == curve_oid:
            break
        if curve_name == _curve_name:
            break
    else:
        if curve_oid:
            raise UnsupportedEccFeature('Unsupported ECC curve (OID: %s)' % curve_oid)
        else:
            raise UnsupportedEccFeature('Unsupported ECC curve (%s)' % curve_name)
    modulus_bytes = curve.p.size_in_bytes()
    point_type = bord(ec_point[0])
    if point_type == 4:
        if len(ec_point) != 1 + 2 * modulus_bytes:
            raise ValueError('Incorrect EC point length')
        x = Integer.from_bytes(ec_point[1:modulus_bytes + 1])
        y = Integer.from_bytes(ec_point[modulus_bytes + 1:])
    elif point_type in (2, 3):
        if len(ec_point) != 1 + modulus_bytes:
            raise ValueError('Incorrect EC point length')
        x = Integer.from_bytes(ec_point[1:])
        y = (x ** 3 - x * 3 + curve.b).sqrt(curve.p)
        if point_type == 2 and y.is_odd():
            y = curve.p - y
        if point_type == 3 and y.is_even():
            y = curve.p - y
    else:
        raise ValueError('Incorrect EC point encoding')
    return construct(curve=_curve_name, point_x=x, point_y=y)