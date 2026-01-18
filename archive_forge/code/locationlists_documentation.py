import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
 Parses a DIE attribute and returns either a LocationExpr or
            a list.
        