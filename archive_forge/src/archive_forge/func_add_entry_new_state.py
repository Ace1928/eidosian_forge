import os
import copy
from collections import namedtuple
from ..common.utils import struct_parse, dwarf_assert
from .constants import *
def add_entry_new_state(cmd, args, is_extended=False):
    entries.append(LineProgramEntry(cmd, is_extended, args, copy.copy(state)))
    state.discriminator = 0
    state.basic_block = False
    state.prologue_end = False
    state.epilogue_begin = False