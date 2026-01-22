import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
def _parse_cie_for_fde(self, fde_offset, fde_header, entry_structs):
    """ Parse the CIE that corresponds to an FDE.
        """
    if self.for_eh_frame:
        cie_displacement = fde_header['CIE_pointer']
        cie_offset = fde_offset + entry_structs.dwarf_format // 8 - cie_displacement
    else:
        cie_offset = fde_header['CIE_pointer']
    with preserve_stream_pos(self.stream):
        return self._parse_entry_at(cie_offset)