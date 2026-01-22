from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
class EndVectorLengthMismatched(RuntimeError):
    """
    The number of elements passed to EndVector does not match the number 
    specified in StartVector.
    """
    pass