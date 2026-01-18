import string
import struct
import time
from numbers import Integral
from ..messages import SPEC_BY_STATUS, Message
from .meta import MetaMessage, build_meta_message, encode_variable_int, meta_charset
from .tracks import MidiTrack, fix_end_of_track, merge_tracks
from .units import tick2second
Prints out all messages in a .midi file.

        May take argument meta_only to show only meta messages.

        Use:
        print_tracks() -> will print all messages
        print_tracks(meta_only=True) -> will print only MetaMessages
        