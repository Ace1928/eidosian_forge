from __future__ import unicode_literals
import binascii
from collections import namedtuple
import six
import struct
import sys
from base64 import urlsafe_b64encode
from pymacaroons.utils import (
from pymacaroons.serializers.base_serializer import BaseSerializer
from pymacaroons.exceptions import MacaroonSerializationException
def _parse_packet_v2(self, data):
    """ Parses a V2 data packet at the start of the given data.

        The format of a packet is as follows:

        field_type(varint) payload_len(varint) data[payload_len bytes]

        apart from EOS which has no payload_en or data (it's a single zero
        byte).

        :param data:
        :return: rest of data, PacketV2
        """
    from pymacaroons.exceptions import MacaroonDeserializationException
    ft, n = _decode_uvarint(data)
    data = data[n:]
    if ft == self._EOS:
        return (data, PacketV2(ft, None))
    payload_len, n = _decode_uvarint(data)
    data = data[n:]
    if payload_len > len(data):
        raise MacaroonDeserializationException('field data extends past end of buffer')
    return (data[payload_len:], PacketV2(ft, data[0:payload_len]))