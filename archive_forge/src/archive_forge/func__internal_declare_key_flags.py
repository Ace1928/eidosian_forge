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
def _internal_declare_key_flags(flag_names, flag_values=_flagvalues.FLAGS, key_flag_values=None):
    """Declares a flag as key for the calling module.

  Internal function.  User code should call declare_key_flag or
  adopt_module_key_flags instead.

  Args:
    flag_names: [str], a list of strings that are names of already-registered
      Flag objects.
    flag_values: FlagValues, the FlagValues instance with which the flags listed
      in flag_names have registered (the value of the flag_values argument from
      the DEFINE_* calls that defined those flags). This should almost never
      need to be overridden.
    key_flag_values: FlagValues, the FlagValues instance that (among possibly
      many other things) keeps track of the key flags for each module. Default
      None means "same as flag_values".  This should almost never need to be
      overridden.

  Raises:
    UnrecognizedFlagError: Raised when the flag is not defined.
  """
    key_flag_values = key_flag_values or flag_values
    module = _helpers.get_calling_module()
    for flag_name in flag_names:
        flag = flag_values[flag_name]
        key_flag_values.register_key_flag_for_module(module, flag)