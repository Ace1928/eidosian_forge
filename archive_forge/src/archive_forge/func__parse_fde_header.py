import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
def _parse_fde_header(self, entry_structs, offset):
    """ Compute a struct to parse the header of the current FDE.
        """
    if not self.for_eh_frame:
        return struct_parse(entry_structs.Dwarf_FDE_header, self.stream, offset)
    fields = [entry_structs.Dwarf_initial_length('length'), entry_structs.Dwarf_offset('CIE_pointer')]
    minimal_header = struct_parse(Struct('eh_frame_minimal_header', *fields), self.stream, offset)
    cie = self._parse_cie_for_fde(offset, minimal_header, entry_structs)
    initial_location_offset = self.stream.tell()
    encoding = cie.augmentation_dict['FDE_encoding']
    assert encoding != DW_EH_encoding_flags['DW_EH_PE_omit']
    basic_encoding = encoding & 15
    encoding_modifier = encoding & 240
    formats = self._eh_encoding_to_field(entry_structs)
    fields.append(formats[basic_encoding]('initial_location'))
    fields.append(formats[basic_encoding]('address_range'))
    result = struct_parse(Struct('Dwarf_FDE_header', *fields), self.stream, offset)
    if encoding_modifier == 0:
        pass
    elif encoding_modifier == DW_EH_encoding_flags['DW_EH_PE_pcrel']:
        result['initial_location'] += self.address + initial_location_offset
    else:
        assert False, 'Unsupported encoding: {:#x}'.format(encoding)
    return result