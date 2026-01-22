import os
import sys
import tempfile
import time
from math import ceil, cos, pi, sin
from types import *
from . import pdfmetrics, pdfutils
from .pdfgeom import bezierArc
from .pdfutils import LINEEND  # this constant needed in both
def MakeFontDictionary(startpos, count):
    """returns a font dictionary assuming they are all in the file from startpos"""
    dict = '  <<' + LINEEND
    pos = startpos
    for i in range(count):
        dict = dict + '\t\t/F%d %d 0 R ' % (i + 1, startpos + i) + LINEEND
    dict = dict + '\t\t>>' + LINEEND
    return dict