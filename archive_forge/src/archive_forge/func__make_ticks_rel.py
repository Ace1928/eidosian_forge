from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
def _make_ticks_rel(self):
    """Make the track's events timing information relative."""
    running_tick = 0
    for event in self.events:
        event.tick -= running_tick
        running_tick += event.tick