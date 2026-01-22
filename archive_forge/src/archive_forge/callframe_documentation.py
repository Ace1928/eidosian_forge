import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
 Decode the instructions contained in the given CFI entry and return
            a DecodedCallFrameTable.
        