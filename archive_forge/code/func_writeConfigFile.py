import datetime
import os
import re
import sys
from collections import OrderedDict
import numpy
from . import units
from .colormap import ColorMap
from .Point import Point
from .Qt import QtCore
def writeConfigFile(data, fname):
    s = genString(data)
    with open(fname, 'wt') as fd:
        fd.write(s)