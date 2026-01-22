import base64
import contextlib
import functools
import hashlib
import struct
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, cast
import dns._features
import dns.exception
import dns.name
import dns.node
import dns.rdata
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.rrset
import dns.transaction
import dns.zone
from dns.dnssectypes import Algorithm, DSDigest, NSEC3Hash
from dns.exception import (  # pylint: disable=W0611
from dns.rdtypes.ANY.CDNSKEY import CDNSKEY
from dns.rdtypes.ANY.CDS import CDS
from dns.rdtypes.ANY.DNSKEY import DNSKEY
from dns.rdtypes.ANY.DS import DS
from dns.rdtypes.ANY.NSEC import NSEC, Bitmap
from dns.rdtypes.ANY.NSEC3PARAM import NSEC3PARAM
from dns.rdtypes.ANY.RRSIG import RRSIG, sigtime_to_posixtime
from dns.rdtypes.dnskeybase import Flag
def _make_cdnskey(public_key: PublicKey, algorithm: Union[int, str], flags: int=Flag.ZONE, protocol: int=3) -> CDNSKEY:
    """Convert a public key to CDNSKEY Rdata

    *public_key*, the public key to convert, a
    ``cryptography.hazmat.primitives.asymmetric`` public key class applicable
    for DNSSEC.

    *algorithm*, a ``str`` or ``int`` specifying the DNSKEY algorithm.

    *flags*: DNSKEY flags field as an integer.

    *protocol*: DNSKEY protocol field as an integer.

    Raises ``ValueError`` if the specified key algorithm parameters are not
    unsupported, ``TypeError`` if the key type is unsupported,
    `UnsupportedAlgorithm` if the algorithm is unknown and
    `AlgorithmKeyMismatch` if the algorithm does not match the key type.

    Return CDNSKEY ``Rdata``.
    """
    dnskey = _make_dnskey(public_key, algorithm, flags, protocol)
    return CDNSKEY(rdclass=dnskey.rdclass, rdtype=dns.rdatatype.CDNSKEY, flags=dnskey.flags, protocol=dnskey.protocol, algorithm=dnskey.algorithm, key=dnskey.key)