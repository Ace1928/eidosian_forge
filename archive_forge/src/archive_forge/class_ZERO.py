import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
class ZERO(object):
    """ End marker for the sequence of CIE/FDE.

    This is specific to `.eh_frame` sections: this kind of entry does not exist
    in pure DWARF. `readelf` displays these as "ZERO terminator", hence the
    class name.
    """

    def __init__(self, offset):
        self.offset = offset