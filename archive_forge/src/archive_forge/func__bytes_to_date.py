import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
def _bytes_to_date(s):
    return date(*time.strptime(s, '%Y-%m-%d')[:3])