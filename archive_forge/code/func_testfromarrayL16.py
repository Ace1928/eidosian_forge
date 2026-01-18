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
def testfromarrayL16(self):
    img = from_array(group(range(2 ** 16), 256), 'L;16')
    img.save('testL16.png')