from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def assertStructIsInline(self, obj):
    """
        Structs are always stored inline, so need to be created right
        where they are used. You'll get this error if you created it
        elsewhere.
        """
    N.enforce_number(obj, N.UOffsetTFlags)
    if obj != self.Offset():
        msg = 'flatbuffers: Tried to write a Struct at an Offset that is different from the current Offset of the Builder.'
        raise StructIsNotInlineError(msg)