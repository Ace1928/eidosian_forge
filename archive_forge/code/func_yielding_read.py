import time
from functools import partial
from io import BytesIO
from itertools import product
from threading import Lock, Thread
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..fileslice import (
def yielding_read(*args, **kwargs):
    time.sleep(0.001)
    return fobj._real_read(*args, **kwargs)