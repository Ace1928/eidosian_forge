from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def EndObject(self):
    """EndObject writes data necessary to finish object construction."""
    self.assertNested()
    self.nested = False
    return self.WriteVtable()