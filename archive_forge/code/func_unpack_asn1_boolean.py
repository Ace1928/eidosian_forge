import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def unpack_asn1_boolean(value: typing.Union[ASN1Value, bytes]) -> bool:
    """Unpacks an ASN.1 BOOLEAN value."""
    b_data = extract_asn1_tlv(value, TagClass.universal, TypeTagNumber.boolean)
    return b_data != b'\x00'