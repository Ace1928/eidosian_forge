from collections import namedtuple, OrderedDict
import os
from ..common.exceptions import DWARFError
from ..common.utils import bytes2str, struct_parse, preserve_stream_pos
from .enums import DW_FORM_raw2name
from .dwarf_util import _resolve_via_offset_table, _get_base_offset
 This is a hook to translate the DW_FORM_...x values in the top DIE
            once the top DIE is parsed to the end. They can't be translated 
            while the top DIE is being parsed, because they implicitly make a
            reference to the DW_AT_xxx_base attribute in the same DIE that may
            not have been parsed yet.
        