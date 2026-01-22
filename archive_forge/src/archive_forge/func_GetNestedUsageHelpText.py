from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
def GetNestedUsageHelpText(field_name, arg_type, required=False):
    """Returns help text for flag with arg_type.

  Generates help text based on schema such that the final output will
  look something like...

    *Foo*
        Required, Foo help text

  Args:
    field_name: str, attribute we are generating help text for
    arg_type: Callable[[str], Any] | None, type of the attribute we are getting
      help text for
    required: bool, whether the attribute is required

  Returns:
    string help text for specific attribute
  """
    if isinstance(arg_type, ArgTypeUsage):
        usage = arg_type.GetUsageHelpText(field_name, required=required)
    else:
        usage = FormatHelpText(field_name=field_name, required=required)
    if usage:
        return '*{}*{}{}'.format(field_name, ASCII_INDENT, IndentAsciiDoc(usage, depth=1))
    else:
        return None