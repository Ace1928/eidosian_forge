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
def _import_der(encoded, passphrase):
    try:
        return _import_subjectPublicKeyInfo(encoded, passphrase)
    except UnsupportedEccFeature as err:
        raise err
    except (ValueError, TypeError, IndexError):
        pass
    try:
        return _import_x509_cert(encoded, passphrase)
    except UnsupportedEccFeature as err:
        raise err
    except (ValueError, TypeError, IndexError):
        pass
    try:
        return _import_rfc5915_der(encoded, passphrase)
    except UnsupportedEccFeature as err:
        raise err
    except (ValueError, TypeError, IndexError):
        pass
    try:
        return _import_pkcs8(encoded, passphrase)
    except UnsupportedEccFeature as err:
        raise err
    except (ValueError, TypeError, IndexError):
        pass
    raise ValueError('Not an ECC DER key')