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
def DEFINE_flag(flag, flag_values=_flagvalues.FLAGS, module_name=None, required=False):
    """Registers a 'Flag' object with a 'FlagValues' object.

  By default, the global FLAGS 'FlagValue' object is used.

  Typical users will use one of the more specialized DEFINE_xxx
  functions, such as DEFINE_string or DEFINE_integer.  But developers
  who need to create Flag objects themselves should use this function
  to register their flags.

  Args:
    flag: Flag, a flag that is key to the module.
    flag_values: FlagValues, the FlagValues instance with which the flag will be
      registered. This should almost never need to be overridden.
    module_name: str, the name of the Python module declaring this flag. If not
      provided, it will be computed using the stack trace of this call.
    required: bool, is this a required flag. This must be used as a keyword
      argument.

  Returns:
    a handle to defined flag.
  """
    if required and flag.default is not None:
        raise ValueError('Required flag --%s cannot have a non-None default' % flag.name)
    fv = flag_values
    fv[flag.name] = flag
    if module_name:
        module = sys.modules.get(module_name)
    else:
        module, module_name = _helpers.get_calling_module_object_and_name()
    flag_values.register_flag_by_module(module_name, flag)
    flag_values.register_flag_by_module_id(id(module), flag)
    if required:
        _validators.mark_flag_as_required(flag.name, fv)
    ensure_non_none_value = flag.default is not None or required
    return _flagvalues.FlagHolder(fv, flag, ensure_non_none_value=ensure_non_none_value)