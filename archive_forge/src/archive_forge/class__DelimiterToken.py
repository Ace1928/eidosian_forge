import binascii
import functools
import logging
import re
import string
import struct
import numpy as np
from matplotlib.cbook import _format_approx
from . import _api
class _DelimiterToken(_Token):
    kind = 'delimiter'

    def is_delim(self):
        return True

    def opposite(self):
        return {'[': ']', ']': '[', '{': '}', '}': '{', '<<': '>>', '>>': '<<'}[self.raw]