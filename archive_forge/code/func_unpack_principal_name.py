import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
def unpack_principal_name(value: typing.Union[bytes, ASN1Value]) -> 'PrincipalName':
    """Unpacks an ASN.1 value to a PrincipalName."""
    s = unpack_asn1_tagged_sequence(value)
    name_type = KerberosPrincipalNameType(get_sequence_value(s, 0, 'PrincipalName', 'name-type', unpack_asn1_integer))
    name = [unpack_asn1_general_string(n) for n in get_sequence_value(s, 1, 'PrincipalName', 'name-string', unpack_asn1_sequence)]
    return PrincipalName(name_type, name)