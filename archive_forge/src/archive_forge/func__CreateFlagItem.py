from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import itertools
import sys
from fire import completion
from fire import custom_descriptions
from fire import decorators
from fire import docstrings
from fire import formatting
from fire import inspectutils
from fire import value_types
def _CreateFlagItem(flag, docstring_info, spec, required=False, flag_string=None, short_arg=False):
    """Returns a string describing a flag using docstring and FullArgSpec info.

  Args:
    flag: The name of the flag.
    docstring_info: A docstrings.DocstringInfo namedtuple with information about
      the containing function's docstring.
    spec: An instance of fire.inspectutils.FullArgSpec, containing type and
     default information about the arguments to a callable.
    required: Whether the flag is required.
    flag_string: If provided, use this string for the flag, rather than
      constructing one from the flag name.
    short_arg: Whether the flag has a short variation or not.
  Returns:
    A string to be used in constructing the help screen for the function.
  """
    max_str_length = LINE_LENGTH - SECTION_INDENTATION - SUBSECTION_INDENTATION
    description = _GetArgDescription(flag, docstring_info)
    if not flag_string:
        flag_string_template = '--{flag_name}={flag_name_upper}'
        flag_string = flag_string_template.format(flag_name=flag, flag_name_upper=formatting.Underline(flag.upper()))
    if required:
        flag_string += ' (required)'
    if short_arg:
        flag_string = '-{short_flag}, '.format(short_flag=flag[0]) + flag_string
    arg_type = _GetArgType(flag, spec)
    arg_default = _GetArgDefault(flag, spec)
    if arg_default == 'None':
        arg_type = 'Optional[{}]'.format(arg_type)
    arg_type = 'Type: {}'.format(arg_type) if arg_type else ''
    available_space = max_str_length - len(arg_type)
    arg_type = formatting.EllipsisTruncate(arg_type, available_space, max_str_length)
    arg_default = 'Default: {}'.format(arg_default) if arg_default else ''
    available_space = max_str_length - len(arg_default)
    arg_default = formatting.EllipsisTruncate(arg_default, available_space, max_str_length)
    description = '\n'.join((part for part in (arg_type, arg_default, description) if part))
    return _CreateItem(flag_string, description, indent=SUBSECTION_INDENTATION)