from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
@thirty_seconds.setter
def thirty_seconds(self, thirty_seconds):
    """
        Set thirty-seconds of the time signature.

        Parameters
        ----------
        thirty_seconds : int
            Thirty-seconds of the time signature.

        """
    self.data[3] = thirty_seconds