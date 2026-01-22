from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
@staticmethod
def _italic_to_slant(italic):
    return FC_SLANT_ITALIC if italic else FC_SLANT_ROMAN