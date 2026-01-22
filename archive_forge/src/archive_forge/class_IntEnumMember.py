import enum
import numpy as np
from .abstract import Dummy, Hashable, Literal, Number, Type
from functools import total_ordering, cached_property
from numba.core import utils
from numba.core.typeconv import Conversion
from numba.np import npdatetime_helpers
class IntEnumMember(EnumMember):
    """
    Type class for IntEnum members.
    """
    basename = 'IntEnum'
    class_type_class = IntEnumClass

    def can_convert_to(self, typingctx, other):
        """
        Convert IntEnum members to plain integers.
        """
        if issubclass(self.instance_class, enum.IntEnum):
            conv = typingctx.can_convert(self.dtype, other)
            return max(conv, Conversion.safe)