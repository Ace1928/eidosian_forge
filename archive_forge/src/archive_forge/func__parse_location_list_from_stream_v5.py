import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
def _parse_location_list_from_stream_v5(self, cu=None):
    """ Returns an array with BaseAddressEntry and LocationEntry.
            No terminator entries.

            The cu argument is necessary if the section is a
            DWARFv5 debug_loclists one, and the target loclist
            contains indirect encodings.
        """
    return [entry_translate[entry.entry_type](entry, cu) for entry in struct_parse(self.structs.Dwarf_loclists_entries, self.stream)]