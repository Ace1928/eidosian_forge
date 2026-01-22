from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class SetTempoEvent(MetaEvent):
    """
    Set Tempo Event.

    """
    meta_command = 81
    length = 3
    name = 'Set Tempo'

    def __str__(self):
        return '%s: tick: %s microseconds per quarter note: %s' % (self.__class__.__name__, self.tick, self.microseconds_per_quarter_note)

    @property
    def microseconds_per_quarter_note(self):
        """
        Microseconds per quarter note.

        """
        assert len(self.data) == 3
        values = [self.data[x] << 16 - 8 * x for x in range(3)]
        return sum(values)

    @microseconds_per_quarter_note.setter
    def microseconds_per_quarter_note(self, microseconds):
        """
        Set microseconds per quarter note.

        Parameters
        ----------
        microseconds : int
            Microseconds per quarter note.

        """
        self.data = [microseconds >> 16 - 8 * x & 255 for x in range(3)]