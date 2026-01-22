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
def _CreateArgItem(arg, docstring_info, spec):
    """Returns a string describing a positional argument.

  Args:
    arg: The name of the positional argument.
    docstring_info: A docstrings.DocstringInfo namedtuple with information about
      the containing function's docstring.
    spec: An instance of fire.inspectutils.FullArgSpec, containing type and
     default information about the arguments to a callable.

  Returns:
    A string to be used in constructing the help screen for the function.
  """
    max_str_length = LINE_LENGTH - SECTION_INDENTATION - SUBSECTION_INDENTATION
    description = _GetArgDescription(arg, docstring_info)
    arg_string = formatting.BoldUnderline(arg.upper())
    arg_type = _GetArgType(arg, spec)
    arg_type = 'Type: {}'.format(arg_type) if arg_type else ''
    available_space = max_str_length - len(arg_type)
    arg_type = formatting.EllipsisTruncate(arg_type, available_space, max_str_length)
    description = '\n'.join((part for part in (arg_type, description) if part))
    return _CreateItem(arg_string, description, indent=SUBSECTION_INDENTATION)