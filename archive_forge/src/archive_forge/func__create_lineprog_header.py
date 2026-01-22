from logging.config import valid_ident
from ..construct import (
from ..common.construct_utils import (RepeatUntilExcluding, ULEB128, SLEB128,
from .enums import *
def _create_lineprog_header(self):
    self.Dwarf_lineprog_file_entry = Struct('file_entry', CString('name'), If(lambda ctx: len(ctx.name) != 0, Embed(Struct('', self.Dwarf_uleb128('dir_index'), self.Dwarf_uleb128('mtime'), self.Dwarf_uleb128('length')))))

    class FormattedEntry(Construct):

        def __init__(self, name, structs, format_field):
            Construct.__init__(self, name)
            self.structs = structs
            self.format_field = format_field

        def _parse(self, stream, context):
            if self.format_field + '_parser' in context:
                parser = context[self.format_field + '_parser']
            else:
                fields = tuple((Rename(f.content_type, self.structs.Dwarf_dw_form[f.form]) for f in context[self.format_field]))
                parser = Struct('formatted_entry', *fields)
                context[self.format_field + '_parser'] = parser
            return parser._parse(stream, context)
    ver5 = lambda ctx: ctx.version >= 5
    self.Dwarf_lineprog_header = Struct('Dwarf_lineprog_header', self.Dwarf_initial_length('unit_length'), self.Dwarf_uint16('version'), If(ver5, self.Dwarf_uint8('address_size'), None), If(ver5, self.Dwarf_uint8('segment_selector_size'), None), self.Dwarf_offset('header_length'), self.Dwarf_uint8('minimum_instruction_length'), If(lambda ctx: ctx.version >= 4, self.Dwarf_uint8('maximum_operations_per_instruction'), 1), self.Dwarf_uint8('default_is_stmt'), self.Dwarf_int8('line_base'), self.Dwarf_uint8('line_range'), self.Dwarf_uint8('opcode_base'), Array(lambda ctx: ctx.opcode_base - 1, self.Dwarf_uint8('standard_opcode_lengths')), If(ver5, PrefixedArray(Struct('directory_entry_format', Enum(self.Dwarf_uleb128('content_type'), **ENUM_DW_LNCT), Enum(self.Dwarf_uleb128('form'), **ENUM_DW_FORM)), self.Dwarf_uint8('directory_entry_format_count'))), If(ver5, PrefixedArray(FormattedEntry('directories', self, 'directory_entry_format'), self.Dwarf_uleb128('directories_count'))), If(ver5, PrefixedArray(Struct('file_name_entry_format', Enum(self.Dwarf_uleb128('content_type'), **ENUM_DW_LNCT), Enum(self.Dwarf_uleb128('form'), **ENUM_DW_FORM)), self.Dwarf_uint8('file_name_entry_format_count'))), If(ver5, PrefixedArray(FormattedEntry('file_names', self, 'file_name_entry_format'), self.Dwarf_uleb128('file_names_count'))), If(lambda ctx: ctx.version < 5, RepeatUntilExcluding(lambda obj, ctx: obj == b'', CString('include_directory'))), If(lambda ctx: ctx.version < 5, RepeatUntilExcluding(lambda obj, ctx: len(obj.name) == 0, self.Dwarf_lineprog_file_entry)))