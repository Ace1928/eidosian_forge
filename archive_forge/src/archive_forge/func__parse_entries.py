import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
def _parse_entries(self):
    entries = []
    offset = 0
    while offset < self.size:
        entries.append(self._parse_entry_at(offset))
        offset = self.stream.tell()
    return entries