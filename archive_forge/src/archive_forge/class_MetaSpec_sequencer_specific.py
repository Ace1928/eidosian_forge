import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class MetaSpec_sequencer_specific(MetaSpec):
    type_byte = 127
    attributes = ['data']
    defaults = [[]]

    def decode(self, message, data):
        message.data = tuple(data)

    def encode(self, message):
        return list(message.data)