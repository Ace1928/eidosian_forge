import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
class SquareWave(NumberGenerator, TimeDependent):
    """
    Generate a square wave with 'on' periods returning 1.0 and
    'off'periods returning 0.0 of specified duration(s). By default
    the portion of time spent in the high state matches the time spent
    in the low state (a duty cycle of 50%), but the duty cycle can be
    controlled if desired.

    The 'on' state begins after a time specified by the 'onset'
    parameter.  The onset duration supplied must be less than the off
    duration.
    """
    onset = param.Number(0.0, doc="Time of onset of the first 'on'\n        state relative to time 0. Must be set to a value less than the\n        'off_duration' parameter.")
    duration = param.Number(1.0, allow_None=False, bounds=(0.0, None), doc="\n         Duration of the 'on' state during which a value of 1.0 is\n         returned.")
    off_duration = param.Number(default=None, allow_None=True, bounds=(0.0, None), doc="\n        Duration of the 'off' value state during which a value of 0.0\n        is returned. By default, this duration matches the value of\n        the 'duration' parameter.")

    def __init__(self, **params):
        super().__init__(**params)
        if self.off_duration is None:
            self.off_duration = self.duration
        if self.onset > self.off_duration:
            raise AssertionError('Onset value needs to be less than %s' % self.onset)

    def __call__(self):
        phase_offset = (self.time_fn() - self.onset) % (self.duration + self.off_duration)
        if phase_offset < self.duration:
            return 1.0
        else:
            return 0.0