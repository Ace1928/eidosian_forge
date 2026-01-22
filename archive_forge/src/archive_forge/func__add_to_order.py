import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
def _add_to_order(regnum):
    if regnum not in reg_order:
        reg_order.append(regnum)