from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def PrependUOffsetTRelativeSlot(self, o, x, d):
    """
        PrependUOffsetTRelativeSlot prepends an UOffsetT onto the object at
        vtable slot `o`. If value `x` equals default `d`, then the slot will
        be set to zero and no other data will be written.
        """
    if x != d or self.forceDefaults:
        self.PrependUOffsetTRelative(x)
        self.Slot(o)