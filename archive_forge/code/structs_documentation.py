from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
Section header parsing.

        Depends on e_machine because of machine-specific values in sh_type.
        