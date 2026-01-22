import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
class VariationInstance(object):

    def __init__(self, name, psname, coords):
        self.name = name
        self.psname = psname
        self.coords = coords

    def __repr__(self):
        return "<VariationInstance '{}' {}>".format(self.name, self.coords)