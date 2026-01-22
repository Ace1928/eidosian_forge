import re
import logging
import warnings
import json
from math import sqrt
import numpy as np
from .base import string_types
from .ndarray import NDArray, load
from . import random
from . import registry
from . import ndarray
from . util import is_np_array
from . import numpy as _mx_np  # pylint: disable=reimported
def _verbose_print(self, desc, init, arr):
    """Internal verbose print function

        Parameters
        ----------
        desc : InitDesc or str
            name of the array
        init : str
            initializer pattern
        arr : NDArray
            initialized array
        """
    if self._verbose and self._print_func:
        logging.info('Initialized %s as %s: %s', desc, init, self._print_func(arr))