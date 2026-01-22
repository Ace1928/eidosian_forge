from . import number_types as N
from .number_types import (UOffsetTFlags, SOffsetTFlags, VOffsetTFlags)
from . import encode
from . import packer
from . import compat
from .compat import range_func
from .compat import memoryview_type
from .compat import import_numpy, NumpyRequiredForThisFeature
import warnings
def CreateSharedString(self, s, encoding='utf-8', errors='strict'):
    """
        CreateSharedString checks if the string is already written to the buffer
        before calling CreateString.
        """
    if s in self.sharedStrings:
        return self.sharedStrings[s]
    off = self.CreateString(s, encoding, errors)
    self.sharedStrings[s] = off
    return off