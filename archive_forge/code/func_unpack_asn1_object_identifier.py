import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def unpack_asn1_object_identifier(value: typing.Union[ASN1Value, bytes]) -> str:
    """Unpacks an ASN.1 OBJECT IDENTIFIER value."""
    b_data = extract_asn1_tlv(value, TagClass.universal, TypeTagNumber.object_identifier)
    first_element = struct.unpack('B', b_data[:1])[0]
    second_element = first_element % 40
    ids = [(first_element - second_element) // 40, second_element]
    idx = 1
    while idx != len(b_data):
        oid, octet_len = _unpack_asn1_octet_number(b_data[idx:])
        ids.append(oid)
        idx += octet_len
    return '.'.join([str(i) for i in ids])