import os
import warnings
from functools import total_ordering
from typing import Union
import numpy as np
class SpacegroupValueError(SpacegroupError):
    """Raised when arguments have invalid value."""
    pass