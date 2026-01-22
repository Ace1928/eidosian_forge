import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def add_flowcontrol_defaults(high, low, kb):
    if high is None:
        if low is None:
            hi = kb * 1024
        else:
            lo = low
            hi = 4 * lo
    else:
        hi = high
    if low is None:
        lo = hi // 4
    else:
        lo = low
    if not hi >= lo >= 0:
        raise ValueError('high (%r) must be >= low (%r) must be >= 0' % (hi, lo))
    return (hi, lo)