from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
@property
def ticks_per_quarter_note(self):
    """
        Number of ticks per quarter note.

        """
    return self.resolution