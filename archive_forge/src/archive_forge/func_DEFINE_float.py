from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import types
from absl.flags import _argument_parser
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _flagvalues
from absl.flags import _helpers
from absl.flags import _validators
def DEFINE_float(name, default, help, lower_bound=None, upper_bound=None, flag_values=_flagvalues.FLAGS, required=False, **args):
    """Registers a flag whose value must be a float.

  If lower_bound or upper_bound are set, then this flag must be
  within the given range.

  Args:
    name: str, the flag name.
    default: float|str|None, the default value of the flag.
    help: str, the help message.
    lower_bound: float, min value of the flag.
    upper_bound: float, max value of the flag.
    flag_values: FlagValues, the FlagValues instance with which the flag will be
      registered. This should almost never need to be overridden.
    required: bool, is this a required flag. This must be used as a keyword
      argument.
    **args: dict, the extra keyword args that are passed to DEFINE.

  Returns:
    a handle to defined flag.
  """
    parser = _argument_parser.FloatParser(lower_bound, upper_bound)
    serializer = _argument_parser.ArgumentSerializer()
    result = DEFINE(parser, name, default, help, flag_values, serializer, required=required, **args)
    _register_bounds_validator_if_needed(parser, name, flag_values=flag_values)
    return result