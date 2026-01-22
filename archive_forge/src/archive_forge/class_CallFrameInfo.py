import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
class CallFrameInfo(object):
    """ DWARF CFI (Call Frame Info)

    Note that this also supports unwinding information as found in .eh_frame
    sections: its format differs slightly from the one in .debug_frame. See
    <http://www.airs.com/blog/archives/460>.

        stream, size:
            A stream holding the .debug_frame section, and the size of the
            section in it.

        address:
            Virtual address for this section. This is used to decode relative
            addresses.

        base_structs:
            The structs to be used as the base for parsing this section.
            Eventually, each entry gets its own structs based on the initial
            length field it starts with. The address_size, however, is taken
            from base_structs. This appears to be a limitation of the DWARFv3
            standard, fixed in v4.
            A discussion I had on dwarf-discuss confirms this.
            So for DWARFv4 we'll take the address size from the CIE header,
            but for earlier versions will use the elfclass of the containing
            file; more sophisticated methods are used by libdwarf and others,
            such as guessing which CU contains which FDEs (based on their
            address ranges) and taking the address_size from those CUs.
    """

    def __init__(self, stream, size, address, base_structs, for_eh_frame=False):
        self.stream = stream
        self.size = size
        self.address = address
        self.base_structs = base_structs
        self.entries = None
        self._entry_cache = {}
        self.for_eh_frame = for_eh_frame

    def get_entries(self):
        """ Get a list of entries that constitute this CFI. The list consists
            of CIE or FDE objects, in the order of their appearance in the
            section.
        """
        if self.entries is None:
            self.entries = self._parse_entries()
        return self.entries

    def _parse_entries(self):
        entries = []
        offset = 0
        while offset < self.size:
            entries.append(self._parse_entry_at(offset))
            offset = self.stream.tell()
        return entries

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

    def _parse_instructions(self, structs, offset, end_offset):
        """ Parse a list of CFI instructions from self.stream, starting with
            the offset and until (not including) end_offset.
            Return a list of CallFrameInstruction objects.
        """
        instructions = []
        while offset < end_offset:
            opcode = struct_parse(structs.Dwarf_uint8(''), self.stream, offset)
            args = []
            primary = opcode & _PRIMARY_MASK
            primary_arg = opcode & _PRIMARY_ARG_MASK
            if primary == DW_CFA_advance_loc:
                args = [primary_arg]
            elif primary == DW_CFA_offset:
                args = [primary_arg, struct_parse(structs.Dwarf_uleb128(''), self.stream)]
            elif primary == DW_CFA_restore:
                args = [primary_arg]
            elif opcode in (DW_CFA_nop, DW_CFA_remember_state, DW_CFA_restore_state, DW_CFA_AARCH64_negate_ra_state):
                args = []
            elif opcode == DW_CFA_set_loc:
                args = [struct_parse(structs.Dwarf_target_addr(''), self.stream)]
            elif opcode == DW_CFA_advance_loc1:
                args = [struct_parse(structs.Dwarf_uint8(''), self.stream)]
            elif opcode == DW_CFA_advance_loc2:
                args = [struct_parse(structs.Dwarf_uint16(''), self.stream)]
            elif opcode == DW_CFA_advance_loc4:
                args = [struct_parse(structs.Dwarf_uint32(''), self.stream)]
            elif opcode in (DW_CFA_offset_extended, DW_CFA_register, DW_CFA_def_cfa, DW_CFA_val_offset):
                args = [struct_parse(structs.Dwarf_uleb128(''), self.stream), struct_parse(structs.Dwarf_uleb128(''), self.stream)]
            elif opcode in (DW_CFA_restore_extended, DW_CFA_undefined, DW_CFA_same_value, DW_CFA_def_cfa_register, DW_CFA_def_cfa_offset):
                args = [struct_parse(structs.Dwarf_uleb128(''), self.stream)]
            elif opcode == DW_CFA_def_cfa_offset_sf:
                args = [struct_parse(structs.Dwarf_sleb128(''), self.stream)]
            elif opcode == DW_CFA_def_cfa_expression:
                args = [struct_parse(structs.Dwarf_dw_form['DW_FORM_block'], self.stream)]
            elif opcode in (DW_CFA_expression, DW_CFA_val_expression):
                args = [struct_parse(structs.Dwarf_uleb128(''), self.stream), struct_parse(structs.Dwarf_dw_form['DW_FORM_block'], self.stream)]
            elif opcode in (DW_CFA_offset_extended_sf, DW_CFA_def_cfa_sf, DW_CFA_val_offset_sf):
                args = [struct_parse(structs.Dwarf_uleb128(''), self.stream), struct_parse(structs.Dwarf_sleb128(''), self.stream)]
            elif opcode == DW_CFA_GNU_args_size:
                args = [struct_parse(structs.Dwarf_uleb128(''), self.stream)]
            else:
                dwarf_assert(False, 'Unknown CFI opcode: 0x%x' % opcode)
            instructions.append(CallFrameInstruction(opcode=opcode, args=args))
            offset = self.stream.tell()
        return instructions

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

    def _parse_cie_augmentation(self, header, entry_structs):
        """ Parse CIE augmentation data from the annotation string in `header`.

        Return a tuple that contains 1) the augmentation data as a string
        (without the length field) and 2) the augmentation data as a dict.
        """
        augmentation = header.get('augmentation')
        if not augmentation:
            return ('', {})
        assert augmentation.startswith(b'z'), 'Unhandled augmentation string: {}'.format(repr(augmentation))
        available_fields = {b'z': entry_structs.Dwarf_uleb128('length'), b'L': entry_structs.Dwarf_uint8('LSDA_encoding'), b'R': entry_structs.Dwarf_uint8('FDE_encoding'), b'S': True, b'P': Struct('personality', entry_structs.Dwarf_uint8('encoding'), Switch('function', lambda ctx: ctx.encoding & 15, {enc: fld_cons('function') for enc, fld_cons in self._eh_encoding_to_field(entry_structs).items()}))}
        fields = []
        aug_dict = {}
        for b in iterbytes(augmentation):
            try:
                fld = available_fields[b]
            except KeyError:
                break
            if fld is True:
                aug_dict[fld] = True
            else:
                fields.append(fld)
        offset = self.stream.tell()
        struct = Struct('Augmentation_Data', *fields)
        aug_dict.update(struct_parse(struct, self.stream, offset))
        self.stream.seek(offset)
        aug_bytes = self._read_augmentation_data(entry_structs)
        return (aug_bytes, aug_dict)

    def _read_augmentation_data(self, entry_structs):
        """ Read augmentation data.

        This assumes that the augmentation string starts with 'z', i.e. that
        augmentation data is prefixed by a length field, which is not returned.
        """
        if not self.for_eh_frame:
            return b''
        augmentation_data_length = struct_parse(Struct('Dummy_Augmentation_Data', entry_structs.Dwarf_uleb128('length')), self.stream)['length']
        return self.stream.read(augmentation_data_length)

    def _parse_lsda_pointer(self, structs, stream_offset, encoding):
        """ Parse bytes to get an LSDA pointer.

        The basic encoding (lower four bits of the encoding) describes how the values are encoded in a CIE or an FDE.
        The modifier (upper four bits of the encoding) describes how the raw values, after decoded using a basic
        encoding, should be modified before using.

        Ref: https://www.airs.com/blog/archives/460
        """
        assert encoding != DW_EH_encoding_flags['DW_EH_PE_omit']
        basic_encoding = encoding & 15
        modifier = encoding & 240
        formats = self._eh_encoding_to_field(structs)
        ptr = struct_parse(Struct('Augmentation_Data', formats[basic_encoding]('LSDA_pointer')), self.stream, stream_pos=stream_offset)['LSDA_pointer']
        if modifier == DW_EH_encoding_flags['DW_EH_PE_absptr']:
            pass
        elif modifier == DW_EH_encoding_flags['DW_EH_PE_pcrel']:
            ptr += self.address + stream_offset
        else:
            assert False, 'Unsupported encoding modifier for LSDA pointer: {:#x}'.format(modifier)
        return ptr

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

    @staticmethod
    def _eh_encoding_to_field(entry_structs):
        """
        Return a mapping from basic encodings (DW_EH_encoding_flags) the
        corresponding field constructors (for instance
        entry_structs.Dwarf_uint32).
        """
        return {DW_EH_encoding_flags['DW_EH_PE_absptr']: entry_structs.Dwarf_target_addr, DW_EH_encoding_flags['DW_EH_PE_uleb128']: entry_structs.Dwarf_uleb128, DW_EH_encoding_flags['DW_EH_PE_udata2']: entry_structs.Dwarf_uint16, DW_EH_encoding_flags['DW_EH_PE_udata4']: entry_structs.Dwarf_uint32, DW_EH_encoding_flags['DW_EH_PE_udata8']: entry_structs.Dwarf_uint64, DW_EH_encoding_flags['DW_EH_PE_sleb128']: entry_structs.Dwarf_sleb128, DW_EH_encoding_flags['DW_EH_PE_sdata2']: entry_structs.Dwarf_int16, DW_EH_encoding_flags['DW_EH_PE_sdata4']: entry_structs.Dwarf_int32, DW_EH_encoding_flags['DW_EH_PE_sdata8']: entry_structs.Dwarf_int64}