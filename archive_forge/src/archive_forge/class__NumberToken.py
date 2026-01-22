import binascii
import functools
import logging
import re
import string
import struct
import numpy as np
from matplotlib.cbook import _format_approx
from . import _api
class _NumberToken(_Token):
    kind = 'number'

    def is_number(self):
        return True

    def value(self):
        if '.' not in self.raw:
            return int(self.raw)
        else:
            return float(self.raw)