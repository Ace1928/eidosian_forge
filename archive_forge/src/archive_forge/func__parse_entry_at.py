import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
def _parse_entry_at(self, offset):
    """ Parse an entry from self.stream starting with the given offset.
            Return the entry object. self.stream will point right after the
            entry.
        """
    if offset in self._entry_cache:
        return self._entry_cache[offset]
    entry_length = struct_parse(self.base_structs.Dwarf_uint32(''), self.stream, offset)
    if self.for_eh_frame and entry_length == 0:
        return ZERO(offset)
    dwarf_format = 64 if entry_length == 4294967295 else 32
    entry_structs = DWARFStructs(little_endian=self.base_structs.little_endian, dwarf_format=dwarf_format, address_size=self.base_structs.address_size)
    CIE_id = struct_parse(entry_structs.Dwarf_offset(''), self.stream)
    if self.for_eh_frame:
        is_CIE = CIE_id == 0
    else:
        is_CIE = dwarf_format == 32 and CIE_id == 4294967295 or CIE_id == 18446744073709551615
    if is_CIE:
        header_struct = entry_structs.EH_CIE_header if self.for_eh_frame else entry_structs.Dwarf_CIE_header
        header = struct_parse(header_struct, self.stream, offset)
    else:
        header = self._parse_fde_header(entry_structs, offset)
    if not self.for_eh_frame and entry_structs.dwarf_version >= 4:
        entry_structs = DWARFStructs(little_endian=entry_structs.little_endian, dwarf_format=entry_structs.dwarf_format, address_size=header.address_size)
    if is_CIE:
        aug_bytes, aug_dict = self._parse_cie_augmentation(header, entry_structs)
    else:
        cie = self._parse_cie_for_fde(offset, header, entry_structs)
        aug_bytes = self._read_augmentation_data(entry_structs)
        lsda_encoding = cie.augmentation_dict.get('LSDA_encoding', DW_EH_encoding_flags['DW_EH_PE_omit'])
        if lsda_encoding != DW_EH_encoding_flags['DW_EH_PE_omit']:
            lsda_pointer = self._parse_lsda_pointer(entry_structs, self.stream.tell() - len(aug_bytes), lsda_encoding)
        else:
            lsda_pointer = None
    end_offset = offset + header.length + entry_structs.initial_length_field_size()
    instructions = self._parse_instructions(entry_structs, self.stream.tell(), end_offset)
    if is_CIE:
        self._entry_cache[offset] = CIE(header=header, instructions=instructions, offset=offset, augmentation_dict=aug_dict, augmentation_bytes=aug_bytes, structs=entry_structs)
    else:
        cie = self._parse_cie_for_fde(offset, header, entry_structs)
        self._entry_cache[offset] = FDE(header=header, instructions=instructions, offset=offset, structs=entry_structs, cie=cie, augmentation_bytes=aug_bytes, lsda_pointer=lsda_pointer)
    return self._entry_cache[offset]