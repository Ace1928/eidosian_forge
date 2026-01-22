from __future__ import print_function
from .utilities import key_number_to_key_name
class PitchBend(object):
    """A pitch bend event.

    Parameters
    ----------
    pitch : int
        MIDI pitch bend amount, in the range ``[-8192, 8191]``.
    time : float
        Time where the pitch bend occurs.

    """

    def __init__(self, pitch, time):
        self.pitch = pitch
        self.time = time

    def __repr__(self):
        return 'PitchBend(pitch={:d}, time={:f})'.format(self.pitch, self.time)