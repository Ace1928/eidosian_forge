import string
import struct
import time
from numbers import Integral
from ..messages import SPEC_BY_STATUS, Message
from .meta import MetaMessage, build_meta_message, encode_variable_int, meta_charset
from .tracks import MidiTrack, fix_end_of_track, merge_tracks
from .units import tick2second
def add_track(self, name=None):
    """Add a new track to the file.

        This will create a new MidiTrack object and append it to the
        track list.
        """
    track = MidiTrack()
    if name is not None:
        track.name = name
    self.tracks.append(track)
    del self.merged_track
    return track