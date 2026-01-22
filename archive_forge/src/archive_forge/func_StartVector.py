from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def StartVector(self, elemSize, numElems, alignment):
    """
        StartVector initializes bookkeeping for writing a new vector.

        A vector has the following format:
          - <UOffsetT: number of elements in this vector>
          - <T: data>+, where T is the type of elements of this vector.
        """
    self.assertNotNested()
    self.nested = True
    self.vectorNumElems = numElems
    self.Prep(N.Uint32Flags.bytewidth, elemSize * numElems)
    self.Prep(alignment, elemSize * numElems)
    return self.Offset()