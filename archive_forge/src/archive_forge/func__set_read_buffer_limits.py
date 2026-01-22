import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _set_read_buffer_limits(self, high=None, low=None):
    high, low = add_flowcontrol_defaults(high, low, constants.FLOW_CONTROL_HIGH_WATER_SSL_READ)
    self._incoming_high_water = high
    self._incoming_low_water = low