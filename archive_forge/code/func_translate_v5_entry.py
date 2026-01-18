import os
from collections import namedtuple
from ..common.utils import struct_parse
from ..common.exceptions import DWARFError
from .dwarf_util import _iter_CUs_in_section
def translate_v5_entry(self, entry, cu):
    """Translates entries in a DWARFv5 rangelist from raw parsed format to 
        a list of BaseAddressEntry/RangeEntry, using the CU
        """
    return entry_translate[entry.entry_type](entry, cu)