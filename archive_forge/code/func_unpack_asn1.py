import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def unpack_asn1(b_data: bytes) -> typing.Tuple[ASN1Value, bytes]:
    """Unpacks an ASN.1 TLV into each element.

    Unpacks the raw ASN.1 value into a `ASN1Value` tuple and returns the remaining bytes that are not part of the
    ASN.1 TLV.

    Args:
        b_data: The raw bytes to unpack as an ASN.1 TLV.

    Returns:
        ASN1Value: The ASN.1 value that is unpacked from the raw bytes passed in.
        bytes: Any remaining bytes that are not part of the ASN1Value.
    """
    octet1 = struct.unpack('B', b_data[:1])[0]
    tag_class = TagClass((octet1 & 192) >> 6)
    constructed = bool(octet1 & 32)
    tag_number = octet1 & 31
    length_offset = 1
    if tag_number == 31:
        tag_number, octet_count = _unpack_asn1_octet_number(b_data[1:])
        length_offset += octet_count
    if tag_class == TagClass.universal:
        tag_number = TypeTagNumber(tag_number)
    b_data = b_data[length_offset:]
    length = struct.unpack('B', b_data[:1])[0]
    length_octets = 1
    if length & 128:
        length_octets += length & 127
        length = 0
        for idx in range(1, length_octets):
            octet_val = struct.unpack('B', b_data[idx:idx + 1])[0]
            length += octet_val << 8 * (length_octets - 1 - idx)
    value = ASN1Value(tag_class=tag_class, constructed=constructed, tag_number=tag_number, b_data=b_data[length_octets:length_octets + length])
    return (value, b_data[length_octets + length:])