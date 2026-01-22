from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
def SetParseFns(*positional, **named):
    """Set the fns for Fire to use to parse args when calling the decorated fn.

  Returns a decorator, which when applied to a function adds metadata to the
  function telling Fire how to turn string command line arguments into proper
  Python arguments with which to call the function.

  A parse function should accept a single string argument and return a value to
  be used in its place when calling the decorated function.

  Args:
    *positional: The functions to be used for parsing positional arguments.
    **named: The functions to be used for parsing named arguments.
  Returns:
    The decorated function, which now has metadata telling Fire how to perform.
  """

    def _Decorator(fn):
        parse_fns = GetParseFns(fn)
        parse_fns['positional'] = positional
        parse_fns['named'].update(named)
        _SetMetadata(fn, FIRE_PARSE_FNS, parse_fns)
        return fn
    return _Decorator