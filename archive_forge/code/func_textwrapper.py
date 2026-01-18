from __future__ import division
import sys
import unicodedata
from functools import reduce
def textwrapper(txt, width):
    return textwrap.wrap(txt, width)