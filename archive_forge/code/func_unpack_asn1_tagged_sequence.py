import collections
import datetime
import enum
import struct
import typing
from spnego._text import to_bytes, to_text
def unpack_asn1_tagged_sequence(value: typing.Union[ASN1Value, bytes]) -> typing.Dict[int, ASN1Value]:
    """Unpacks an ASN.1 SEQUENCE value as a dictionary."""
    return dict([(e.tag_number, unpack_asn1(e.b_data)[0]) for e in unpack_asn1_sequence(value)])