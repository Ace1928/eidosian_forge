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
def tick_to_time(self, tick):
    """Converts from an absolute tick to time in seconds using
        ``self.__tick_to_time``.

        Parameters
        ----------
        tick : int
            Absolute tick to convert.

        Returns
        -------
        time : float
            Time in seconds of tick.

        """
    if tick >= MAX_TICK:
        raise IndexError('Supplied tick is too large.')
    if tick >= len(self.__tick_to_time):
        self._update_tick_to_time(tick)
    if not isinstance(tick, int):
        warnings.warn('tick should be an int.')
    return self.__tick_to_time[int(tick)]