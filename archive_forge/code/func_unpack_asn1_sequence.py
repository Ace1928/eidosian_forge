import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def unpack_asn1_sequence(value: typing.Union[ASN1Value, bytes]) -> typing.List[ASN1Value]:
    """Unpacks an ASN.1 SEQUENCE value."""
    b_data = extract_asn1_tlv(value, TagClass.universal, TypeTagNumber.sequence)
    values = []
    while b_data:
        v, b_data = unpack_asn1(b_data)
        values.append(v)
    return values