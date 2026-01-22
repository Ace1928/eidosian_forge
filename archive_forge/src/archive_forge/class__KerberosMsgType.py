import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class _KerberosMsgType(type):
    __registry: typing.Dict[int, typing.Dict[int, '_KerberosMsgType']] = {}

    def __init__(cls, *args: typing.Any, **kwargs: typing.Any) -> None:
        pvno = getattr(cls, 'PVNO', 0)
        if pvno not in cls.__registry:
            cls.__registry[pvno] = {}
        msg_type = getattr(cls, 'MESSAGE_TYPE', None)
        if msg_type is not None:
            cls.__registry[pvno][msg_type] = cls

    def __call__(cls, sequence: typing.Dict[int, ASN1Value]) -> '_KerberosMsgType':
        pvno_idx = 0
        if 0 not in sequence:
            pvno_idx = 1
        pvno = unpack_asn1_integer(sequence[pvno_idx])
        message_type = unpack_asn1_integer(sequence[pvno_idx + 1])
        new_cls = cls.__registry[pvno].get(message_type, cls)
        return super(_KerberosMsgType, new_cls).__call__(sequence)