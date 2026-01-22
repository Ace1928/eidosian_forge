from numpy.ma import (
import numpy.ma as ma
import warnings
import numpy as np
from numpy import (
from numpy.core.records import (
@property
def _fieldmask(self):
    """
        Alias to mask.

        """
    return self._mask