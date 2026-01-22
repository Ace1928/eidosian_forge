from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
class MetaEventWithText(MetaEvent):
    """
    Meta Event With Text.

    """

    def __init__(self, **kwargs):
        super(MetaEventWithText, self).__init__(**kwargs)
        if 'text' not in kwargs:
            self.text = ''.join((chr(datum) for datum in self.data))

    def __str__(self):
        return '%s: %s' % (self.__class__.__name__, self.text)