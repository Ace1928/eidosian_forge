from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
class PruneStats(_prunestats):
    """ Holds statistics from reference count pruning.
    """

    def __add__(self, other):
        if not isinstance(other, PruneStats):
            msg = 'PruneStats can only be added to another PruneStats, got {}.'
            raise TypeError(msg.format(type(other)))
        return PruneStats(self.basicblock + other.basicblock, self.diamond + other.diamond, self.fanout + other.fanout, self.fanout_raise + other.fanout_raise)

    def __sub__(self, other):
        if not isinstance(other, PruneStats):
            msg = 'PruneStats can only be subtracted from another PruneStats, got {}.'
            raise TypeError(msg.format(type(other)))
        return PruneStats(self.basicblock - other.basicblock, self.diamond - other.diamond, self.fanout - other.fanout, self.fanout_raise - other.fanout_raise)