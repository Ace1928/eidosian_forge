import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
def unpack_ticket(value: typing.Union[ASN1Value, bytes]) -> typing.List[Ticket]:
    return [Ticket.unpack(t) for t in unpack_asn1_sequence(value)]