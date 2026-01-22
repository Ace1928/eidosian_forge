from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import inspect
from fire import inspectutils
import six
def _CompletionsFromArgs(fn_args):
    """Takes a list of fn args and returns a list of the fn's completion strings.

  Args:
    fn_args: A list of the args accepted by a function.
  Returns:
    A list of possible completion strings for that function.
  """
    completions = []
    for arg in fn_args:
        arg = arg.replace('_', '-')
        completions.append('--{arg}'.format(arg=arg))
    return completions