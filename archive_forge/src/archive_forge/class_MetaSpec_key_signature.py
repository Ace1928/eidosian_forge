import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class MetaSpec_key_signature(MetaSpec):
    type_byte = 89
    attributes = ['key']
    defaults = ['C']

    def decode(self, message, data):
        key = signed('byte', data[0])
        mode = data[1]
        try:
            message.key = _key_signature_decode[key, mode]
        except KeyError as ke:
            if key < 7:
                msg = 'Could not decode key with {} flats and mode {}'.format(abs(key), mode)
            else:
                msg = 'Could not decode key with {} sharps and mode {}'.format(key, mode)
            raise KeySignatureError(msg) from ke

    def encode(self, message):
        key, mode = _key_signature_encode[message.key]
        return [unsigned('byte', key), mode]

    def check(self, name, value):
        if value not in _key_signature_encode:
            raise ValueError(f'invalid key {value!r}')