from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class AfterTouchEvent(ChannelEvent):
    """
    After Touch Event.

    """
    status_msg = 160
    length = 2
    name = 'After Touch'

    def __str__(self):
        return '%s: tick: %s channel: %s pitch: %s value: %s' % (self.__class__.__name__, self.tick, self.channel, self.pitch, self.value)

    @property
    def pitch(self):
        """
        Pitch of the after touch event.

        """
        return self.data[0]

    @pitch.setter
    def pitch(self, pitch):
        """
        Set the pitch of the after touch event.

        Parameters
        ----------
        pitch : int
            Pitch of the after touch event.

        """
        self.data[0] = pitch

    @property
    def value(self):
        """
        Value of the after touch event.

        """
        return self.data[1]

    @value.setter
    def value(self, value):
        """
        Set the value of the after touch event.

        Parameters
        ----------
        value : int
            Value of the after touch event.

        """
        self.data[1] = value