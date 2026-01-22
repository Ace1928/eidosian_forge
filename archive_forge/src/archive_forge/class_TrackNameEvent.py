from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class TrackNameEvent(MetaEventWithText):
    """
    Track Name Event.

    """
    meta_command = 3
    length = 'variable'
    name = 'Track Name'