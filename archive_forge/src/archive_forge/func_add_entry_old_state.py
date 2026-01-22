import os
import copy
from collections import namedtuple
from ..common.utils import struct_parse, dwarf_assert
from .constants import *
def add_entry_old_state(cmd, args, is_extended=False):
    entries.append(LineProgramEntry(cmd, is_extended, args, None))