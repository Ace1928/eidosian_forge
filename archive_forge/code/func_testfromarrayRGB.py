from the public API.  This format is called packed.  When packed,
import io
import itertools
import math
import operator
import struct
import sys
import zlib
import warnings
from array import array
from functools import reduce
from pygame.tests.test_utils import tostring
fromarray = from_array
import tempfile
import unittest
def testfromarrayRGB(self):
    img = from_array([[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1], [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]], 'RGB;1')
    o = BytesIO()
    img.save(o)