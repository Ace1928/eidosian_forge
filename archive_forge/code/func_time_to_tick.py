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
def time_to_tick(self, time):
    """Converts from a time in seconds to absolute tick using
        ``self._tick_scales``.

        Parameters
        ----------
        time : float
            Time, in seconds.

        Returns
        -------
        tick : int
            Absolute tick corresponding to the supplied time.

        """
    tick = np.searchsorted(self.__tick_to_time, time, side='left')
    if tick == len(self.__tick_to_time):
        tick -= 1
        _, final_tick_scale = self._tick_scales[-1]
        tick += (time - self.__tick_to_time[tick]) / final_tick_scale
        return int(round(tick))
    if tick and math.fabs(time - self.__tick_to_time[tick - 1]) < math.fabs(time - self.__tick_to_time[tick]):
        return tick - 1
    else:
        return tick