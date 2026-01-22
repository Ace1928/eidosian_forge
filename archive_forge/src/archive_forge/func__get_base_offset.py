import os
from ..construct.macros import UBInt32, UBInt64, ULInt32, ULInt64, Array
from ..common.exceptions import DWARFError
from ..common.utils import preserve_stream_pos, struct_parse
def _get_base_offset(cu, base_attribute_name):
    """Retrieves a required, base offset-type atribute
    from the top DIE in the CU. Applies to several indirectly
    encoded objects - range lists, location lists, strings, addresses.
    """
    cu_top_die = cu.get_top_DIE()
    if not base_attribute_name in cu_top_die.attributes:
        raise DWARFError('The CU at offset 0x%x needs %s' % (cu.cu_offset, base_attribute_name))
    return cu_top_die.attributes[base_attribute_name].value