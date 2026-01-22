import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class MetaSpec_midi_port(MetaSpec):
    type_byte = 33
    attributes = ['port']
    defaults = [0]

    def decode(self, message, data):
        if len(data) == 0:
            message.port = 0
        else:
            message.port = data[0]

    def encode(self, message):
        return [message.port]

    def check(self, name, value):
        check_int(value, 0, 255)