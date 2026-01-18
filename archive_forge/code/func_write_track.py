import string
import struct
import time
from numbers import Integral
from ..messages import SPEC_BY_STATUS, Message
from .meta import MetaMessage, build_meta_message, encode_variable_int, meta_charset
from .tracks import MidiTrack, fix_end_of_track, merge_tracks
from .units import tick2second
def write_track(outfile, track):
    data = bytearray()
    running_status_byte = None
    for msg in fix_end_of_track(track):
        if not isinstance(msg.time, Integral):
            raise ValueError('message time must be int in MIDI file')
        if msg.time < 0:
            raise ValueError('message time must be non-negative in MIDI file')
        if msg.is_realtime:
            raise ValueError('realtime messages are not allowed in MIDI files')
        data.extend(encode_variable_int(msg.time))
        if msg.is_meta:
            data.extend(msg.bytes())
            running_status_byte = None
        elif msg.type == 'sysex':
            data.append(240)
            data.extend(encode_variable_int(len(msg.data) + 1))
            data.extend(msg.data)
            data.append(247)
            running_status_byte = None
        else:
            msg_bytes = msg.bytes()
            status_byte = msg_bytes[0]
            if status_byte == running_status_byte:
                data.extend(msg_bytes[1:])
            else:
                data.extend(msg_bytes)
            if status_byte < 240:
                running_status_byte = status_byte
            else:
                running_status_byte = None
    write_chunk(outfile, b'MTrk', data)