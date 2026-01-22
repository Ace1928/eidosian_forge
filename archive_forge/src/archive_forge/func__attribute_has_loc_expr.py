import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
@staticmethod
def _attribute_has_loc_expr(attr, dwarf_version):
    return dwarf_version < 4 and attr.form.startswith('DW_FORM_block') and (not attr.name == 'DW_AT_const_value') or attr.form == 'DW_FORM_exprloc'