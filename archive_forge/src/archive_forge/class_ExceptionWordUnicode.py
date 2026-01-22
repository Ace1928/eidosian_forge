import re
import sys
import typing
from .util import (
from .unicode import pyparsing_unicode as ppu
class ExceptionWordUnicode(ppu.Latin1, ppu.LatinA, ppu.LatinB, ppu.Greek, ppu.Cyrillic):
    pass