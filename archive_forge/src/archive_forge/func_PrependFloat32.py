from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def PrependFloat32(self, x):
    """Prepend a `float32` to the Builder buffer.

        Note: aligns and checks for space.
        """
    self.Prepend(N.Float32Flags, x)