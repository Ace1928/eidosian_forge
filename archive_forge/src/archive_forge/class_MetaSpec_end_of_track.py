import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class MetaSpec_end_of_track(MetaSpec):
    type_byte = 47
    attributes = []
    defaults = []

    def decode(self, message, data):
        pass

    def encode(self, message):
        return []