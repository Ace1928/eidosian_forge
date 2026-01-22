from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def PrependStructSlot(self, v, x, d):
    """
        PrependStructSlot prepends a struct onto the object at vtable slot `o`.
        Structs are stored inline, so nothing additional is being added.
        In generated code, `d` is always 0.
        """
    N.enforce_number(d, N.UOffsetTFlags)
    if x != d:
        self.assertStructIsInline(x)
        self.Slot(v)