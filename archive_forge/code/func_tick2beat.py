from __future__ import absolute_import, division, print_function
import numpy as np
import mido
def tick2beat(tick, ticks_per_beat=DEFAULT_TICKS_PER_BEAT, time_signature=DEFAULT_TIME_SIGNATURE):
    """
    Convert ticks to beats.

    Returns beats for a chosen MIDI file time resolution (ticks/pulses per
    quarter note, also called PPQN) and time signature.

    """
    return tick / (4.0 * ticks_per_beat / time_signature[1])