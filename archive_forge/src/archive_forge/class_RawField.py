from Xlib import X
from Xlib.protocol import rq
class RawField(rq.ValueField):
    """A field with raw data, stored as a string"""
    structcode = None

    def pack_value(self, val):
        return (val, len(val), None)

    def parse_binary_value(self, data, display, length, format):
        return (data, '')