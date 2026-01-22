from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import io
import itertools
import json
import sys
import wcwidth
@staticmethod
def Abbreviate(s, width):
    """Abbreviate a string to at most width characters."""
    suffix = '.' * min(width, 3)
    return s if len(s) <= width else s[:width - len(suffix)] + suffix