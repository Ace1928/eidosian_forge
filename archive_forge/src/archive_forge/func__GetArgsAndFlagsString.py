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
def _GetArgsAndFlagsString(spec, metadata):
    """The args and flags string for showing how to call a function.

  If positional arguments are accepted, the args will be shown as positional.
  E.g. "ARG1 ARG2 [--flag=FLAG]"

  If positional arguments are disallowed, the args will be shown with flags
  syntax.
  E.g. "--arg1=ARG1 [--flag=FLAG]"

  Args:
    spec: The full arg spec for the component to construct the args and flags
      string for.
    metadata: Metadata for the component, including whether it accepts
      positional arguments.

  Returns:
    The constructed args and flags string.
  """
    args_with_no_defaults = spec.args[:len(spec.args) - len(spec.defaults)]
    args_with_defaults = spec.args[len(spec.args) - len(spec.defaults):]
    accepts_positional_args = metadata.get(decorators.ACCEPTS_POSITIONAL_ARGS)
    arg_and_flag_strings = []
    if args_with_no_defaults:
        if accepts_positional_args:
            arg_strings = [formatting.Underline(arg.upper()) for arg in args_with_no_defaults]
        else:
            arg_strings = ['--{arg}={arg_upper}'.format(arg=arg, arg_upper=formatting.Underline(arg.upper())) for arg in args_with_no_defaults]
        arg_and_flag_strings.extend(arg_strings)
    if args_with_defaults or spec.kwonlyargs or spec.varkw:
        arg_and_flag_strings.append('<flags>')
    if spec.varargs:
        varargs_string = '[{varargs}]...'.format(varargs=formatting.Underline(spec.varargs.upper()))
        arg_and_flag_strings.append(varargs_string)
    return ' '.join(arg_and_flag_strings)