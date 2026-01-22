from __future__ import division         # use "true" division instead of integer division in Python 2 (see PEP 238)
from __future__ import print_function   # use print() as a function in Python 2 (see PEP 3105)
from __future__ import absolute_import  # use absolute imports by default in Python 2 (see PEP 328)
import math
import optparse
import os
import re
import sys
import time
import xml.dom.minidom
from xml.dom import Node, NotFoundErr
from collections import namedtuple, defaultdict
from decimal import Context, Decimal, InvalidOperation, getcontext
import six
from six.moves import range, urllib
from scour.svg_regex import svg_parser
from scour.svg_transform import svg_transform_parser
from scour.yocto_css import parseCssString
from scour import __version__
class SVGLength(object):

    def __init__(self, str):
        try:
            self.value = float(str)
            if int(self.value) == self.value:
                self.value = int(self.value)
            self.units = Unit.NONE
        except ValueError:
            self.value = 0
            unitBegin = 0
            scinum = scinumber.match(str)
            if scinum is not None:
                numMatch = number.match(str)
                expMatch = sciExponent.search(str, numMatch.start(0))
                self.value = float(numMatch.group(0)) * 10 ** float(expMatch.group(1))
                unitBegin = expMatch.end(1)
            else:
                numMatch = number.match(str)
                if numMatch is not None:
                    self.value = float(numMatch.group(0))
                    unitBegin = numMatch.end(0)
            if int(self.value) == self.value:
                self.value = int(self.value)
            if unitBegin != 0:
                unitMatch = unit.search(str, unitBegin)
                if unitMatch is not None:
                    self.units = Unit.get(unitMatch.group(0))
            else:
                self.value = 0
                self.units = Unit.INVALID