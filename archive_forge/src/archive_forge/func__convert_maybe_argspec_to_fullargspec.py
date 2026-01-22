import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def _convert_maybe_argspec_to_fullargspec(argspec):
    if isinstance(argspec, FullArgSpec):
        return argspec
    return FullArgSpec(args=argspec.args, varargs=argspec.varargs, varkw=argspec.keywords, defaults=argspec.defaults, kwonlyargs=[], kwonlydefaults=None, annotations={})