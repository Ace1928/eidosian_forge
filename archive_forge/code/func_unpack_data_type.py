import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
def unpack_data_type(value: typing.Union[ASN1Value, bytes]) -> typing.Union[KerberosPADataType, int]:
    int_val = unpack_asn1_integer(value)
    try:
        return KerberosPADataType(int_val)
    except ValueError:
        return int_val