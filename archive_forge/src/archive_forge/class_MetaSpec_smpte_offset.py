import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class MetaSpec_smpte_offset(MetaSpec):
    type_byte = 84
    attributes = ['frame_rate', 'hours', 'minutes', 'seconds', 'frames', 'sub_frames']
    defaults = [24, 0, 0, 0, 0, 0]

    def decode(self, message, data):
        message.frame_rate = _smpte_framerate_decode[data[0] >> 5]
        message.hours = data[0] & 31
        message.minutes = data[1]
        message.seconds = data[2]
        message.frames = data[3]
        message.sub_frames = data[4]

    def encode(self, message):
        frame_rate_lookup = _smpte_framerate_encode[message.frame_rate] << 5
        return [frame_rate_lookup | message.hours, message.minutes, message.seconds, message.frames, message.sub_frames]

    def check(self, name, value):
        if name == 'frame_rate':
            if value not in _smpte_framerate_encode:
                valid = ', '.join(sorted(_smpte_framerate_encode.keys()))
                raise ValueError(f'frame_rate must be one of {valid}')
        elif name == 'hours':
            check_int(value, 0, 255)
        elif name in ['minutes', 'seconds']:
            check_int(value, 0, 59)
        elif name == 'frames':
            check_int(value, 0, 255)
        elif name == 'sub_frames':
            check_int(value, 0, 99)