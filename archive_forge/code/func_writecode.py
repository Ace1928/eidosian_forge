from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr
from fontTools.misc.textTools import byteord, safeEval
from . import DefaultTable
from . import grUtils
from array import array
from functools import reduce
import struct, re, sys
def writecode(tag, writer, instrs):
    writer.begintag(tag)
    writer.newline()
    for l in disassemble(instrs):
        writer.write(l)
        writer.newline()
    writer.endtag(tag)
    writer.newline()