import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class MetaSpec_track_name(MetaSpec_text):
    type_byte = 3
    attributes = ['name']
    defaults = ['']

    def decode(self, message, data):
        message.name = decode_string(data)

    def encode(self, message):
        return encode_string(message.name)