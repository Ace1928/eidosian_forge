import math
import numpy as np
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import tensor_format
from tensorflow.python.debug.lib import common
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
def time_to_readable_str(value_us, force_time_unit=None):
    """Convert time value to human-readable string.

  Args:
    value_us: time value in microseconds.
    force_time_unit: force the output to use the specified time unit. Must be
      in TIME_UNITS.

  Returns:
    Human-readable string representation of the time value.

  Raises:
    ValueError: if force_time_unit value is not in TIME_UNITS.
  """
    if not value_us:
        return '0'
    if force_time_unit:
        if force_time_unit not in TIME_UNITS:
            raise ValueError('Invalid time unit: %s' % force_time_unit)
        order = TIME_UNITS.index(force_time_unit)
        time_unit = force_time_unit
        return '{:.10g}{}'.format(value_us / math.pow(10.0, 3 * order), time_unit)
    else:
        order = min(len(TIME_UNITS) - 1, int(math.log(value_us, 10) / 3))
        time_unit = TIME_UNITS[order]
        return '{:.3g}{}'.format(value_us / math.pow(10.0, 3 * order), time_unit)