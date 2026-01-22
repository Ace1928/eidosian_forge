import os
from collections import namedtuple
from ..common.utils import struct_parse
from bisect import bisect_right
import math
def _get_addr_size_struct(self, addr_header_value):
    """ Given this set's header value (int) for the address size,
            get the Construct representation of that size
        """
    if addr_header_value == 4:
        return self.structs.Dwarf_uint32
    else:
        assert addr_header_value == 8
        return self.structs.Dwarf_uint64