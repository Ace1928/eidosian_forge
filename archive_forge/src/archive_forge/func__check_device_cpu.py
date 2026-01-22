import itertools
import math
from functools import wraps
import numpy
import scipy.special as special
from .._config import get_config
from .fixes import parse_version
def _check_device_cpu(device):
    if device not in {'cpu', None}:
        raise ValueError(f'Unsupported device for NumPy: {device!r}')