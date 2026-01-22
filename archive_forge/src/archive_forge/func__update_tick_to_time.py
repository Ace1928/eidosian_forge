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
def _update_tick_to_time(self, max_tick):
    """Creates ``self.__tick_to_time``, a class member array which maps
        ticks to time starting from tick 0 and ending at ``max_tick``.

        Parameters
        ----------
        max_tick : int
            Last tick to compute time for.  If ``self._tick_scales`` contains a
            tick which is larger than this value, it will be used instead.

        """
    max_scale_tick = max((ts[0] for ts in self._tick_scales))
    max_tick = max_tick if max_tick > max_scale_tick else max_scale_tick
    self.__tick_to_time = np.zeros(max_tick + 1)
    last_end_time = 0
    for (start_tick, tick_scale), (end_tick, _) in zip(self._tick_scales[:-1], self._tick_scales[1:]):
        ticks = np.arange(end_tick - start_tick + 1)
        self.__tick_to_time[start_tick:end_tick + 1] = last_end_time + tick_scale * ticks
        last_end_time = self.__tick_to_time[end_tick]
    start_tick, tick_scale = self._tick_scales[-1]
    ticks = np.arange(max_tick + 1 - start_tick)
    self.__tick_to_time[start_tick:] = last_end_time + tick_scale * ticks