from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
import types
from fire import docstrings
import six
def Py3GetFullArgSpec(fn):
    """A alternative to the builtin getfullargspec.

  The builtin inspect.getfullargspec uses:
  `skip_bound_args=False, follow_wrapped_chains=False`
  in order to be backwards compatible.

  This function instead skips bound args (self) and follows wrapped chains.

  Args:
    fn: The function or class of interest.
  Returns:
    An inspect.FullArgSpec namedtuple with the full arg spec of the function.
  """
    try:
        sig = inspect._signature_from_callable(fn, skip_bound_arg=True, follow_wrapper_chains=True, sigcls=inspect.Signature)
    except Exception:
        raise TypeError('Unsupported callable.')
    args = []
    varargs = None
    varkw = None
    kwonlyargs = []
    defaults = ()
    annotations = {}
    defaults = ()
    kwdefaults = {}
    if sig.return_annotation is not sig.empty:
        annotations['return'] = sig.return_annotation
    for param in sig.parameters.values():
        kind = param.kind
        name = param.name
        if kind is inspect._POSITIONAL_ONLY:
            args.append(name)
        elif kind is inspect._POSITIONAL_OR_KEYWORD:
            args.append(name)
            if param.default is not param.empty:
                defaults += (param.default,)
        elif kind is inspect._VAR_POSITIONAL:
            varargs = name
        elif kind is inspect._KEYWORD_ONLY:
            kwonlyargs.append(name)
            if param.default is not param.empty:
                kwdefaults[name] = param.default
        elif kind is inspect._VAR_KEYWORD:
            varkw = name
        if param.annotation is not param.empty:
            annotations[name] = param.annotation
    if not kwdefaults:
        kwdefaults = None
    if not defaults:
        defaults = None
    return inspect.FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwdefaults, annotations)