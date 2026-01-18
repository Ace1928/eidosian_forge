import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
def unpack_kdc_options(value: typing.Union[ASN1Value, bytes]) -> int:
    b_data = unpack_asn1_bit_string(value)
    return struct.unpack('>I', b_data)[0]