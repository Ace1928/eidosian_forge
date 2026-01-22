import string
import struct
import time
from numbers import Integral
from ..messages import SPEC_BY_STATUS, Message
from .meta import MetaMessage, build_meta_message, encode_variable_int, meta_charset
from .tracks import MidiTrack, fix_end_of_track, merge_tracks
from .units import tick2second
class DebugFileWrapper:

    def __init__(self, file):
        self.file = file

    def read(self, size):
        data = self.file.read(size)
        for byte in data:
            print_byte(byte, self.file.tell())
        return data

    def tell(self):
        return self.file.tell()