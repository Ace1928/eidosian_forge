from collections import namedtuple, OrderedDict
import os
from ..common.exceptions import DWARFError
from ..common.utils import bytes2str, struct_parse, preserve_stream_pos
from .enums import DW_FORM_raw2name
from .dwarf_util import _resolve_via_offset_table, _get_base_offset
def _translate_indirect_attributes(self):
    """ This is a hook to translate the DW_FORM_...x values in the top DIE
            once the top DIE is parsed to the end. They can't be translated 
            while the top DIE is being parsed, because they implicitly make a
            reference to the DW_AT_xxx_base attribute in the same DIE that may
            not have been parsed yet.
        """
    for key in self.attributes:
        attr = self.attributes[key]
        if attr.form in ('DW_FORM_strx', 'DW_FORM_strx1', 'DW_FORM_strx2', 'DW_FORM_strx3', 'DW_FORM_strx4', 'DW_FORM_addrx', 'DW_FORM_addrx1', 'DW_FORM_addrx2', 'DW_FORM_addrx3', 'DW_FORM_addrx4', 'DW_FORM_loclistx', 'DW_FORM_rnglistx'):
            self.attributes[key] = AttributeValue(name=attr.name, form=attr.form, value=self._translate_attr_value(attr.form, attr.raw_value), raw_value=attr.raw_value, offset=attr.offset, indirection_length=attr.indirection_length)