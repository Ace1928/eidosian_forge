import os
import copy
from collections import namedtuple
from ..common.utils import struct_parse, dwarf_assert
from .constants import *
class LineState(object):
    """ Represents a line program state (or a "row" in the matrix
        describing debug location information for addresses).
        The instance variables of this class are the "state machine registers"
        described in section 6.2.2 of DWARFv3
    """

    def __init__(self, default_is_stmt):
        self.address = 0
        self.file = 1
        self.line = 1
        self.column = 0
        self.op_index = 0
        self.is_stmt = default_is_stmt
        self.basic_block = False
        self.end_sequence = False
        self.prologue_end = False
        self.epilogue_begin = False
        self.isa = 0
        self.discriminator = 0

    def __repr__(self):
        a = ['<LineState %x:' % id(self)]
        a.append('  address = 0x%x' % self.address)
        for attr in ('file', 'line', 'column', 'is_stmt', 'basic_block', 'end_sequence', 'prologue_end', 'epilogue_begin', 'isa', 'discriminator'):
            a.append('  %s = %s' % (attr, getattr(self, attr)))
        return '\n'.join(a) + '>\n'