from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
import types
from fire import docstrings
import six
def GetFullArgSpec(fn):
    """Returns a FullArgSpec describing the given callable."""
    original_fn = fn
    fn, skip_arg = _GetArgSpecInfo(fn)
    try:
        if sys.version_info[0:2] >= (3, 5):
            args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = Py3GetFullArgSpec(fn)
        elif six.PY3:
            args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(fn)
        else:
            args, varargs, varkw, defaults = Py2GetArgSpec(fn)
            kwonlyargs = kwonlydefaults = None
            annotations = getattr(fn, '__annotations__', None)
    except TypeError:
        if inspect.isbuiltin(fn):
            return FullArgSpec(varargs='vars', varkw='kwargs')
        fields = getattr(original_fn, '_fields', None)
        if fields is not None:
            return FullArgSpec(args=list(fields))
        return FullArgSpec()
    skip_arg_required = six.PY2 or sys.version_info[0:2] == (3, 4)
    if skip_arg_required and skip_arg and args:
        args.pop(0)
    return FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations)