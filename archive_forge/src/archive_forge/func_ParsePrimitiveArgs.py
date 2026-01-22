from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import functools
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from six.moves import map  # pylint: disable=redefined-builtin
def ParsePrimitiveArgs(args, arg_name, current_value_thunk):
    """Parse the modification to the given repeated field.

  To be used in combination with AddPrimitiveArgs; see module docstring.

  Args:
    args: argparse.Namespace of parsed arguments
    arg_name: string, the (plural) suffix of the argument (snake_case).
    current_value_thunk: zero-arg function that returns the current value of the
      attribute to be updated. Will be called lazily if required.

  Raises:
    ValueError: if more than one arg is set.

  Returns:
    List of str: the new value for the field, or None if no change is required.

  """
    underscored_name = arg_name.replace('-', '_')
    remove = getattr(args, 'remove_' + underscored_name)
    add = getattr(args, 'add_' + underscored_name)
    clear = getattr(args, 'clear_' + underscored_name)
    set_ = getattr(args, 'set_' + underscored_name, None)
    return _ModifyCurrentValue(remove, add, clear, set_, current_value_thunk)