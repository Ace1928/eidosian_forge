from __future__ import print_function
import mido
import numpy as np
import math
import warnings
import collections
import copy
import functools
import six
from heapq import merge
from .instrument import Instrument
from .containers import (KeySignature, TimeSignature, Lyric, Note,
from .utilities import (key_name_to_key_number, qpm_to_bpm)
def adjust_meta(events):
    """ This function adjusts the timing of the track-level meta-events
            in the provided list"""
    events.sort(key=lambda e: e.time)
    event_times = np.array([event.time for event in events])
    adjusted_event_times = np.interp(event_times, original_times, new_times)
    for event, adjusted_event_time in zip(events, adjusted_event_times):
        event.time = adjusted_event_time
    valid_events = [event for event in events if event.time == new_times[0]]
    if valid_events:
        valid_events = valid_events[-1:]
    valid_events.extend((event for event in events if event.time > new_times[0] and event.time < new_times[-1]))
    events[:] = valid_events