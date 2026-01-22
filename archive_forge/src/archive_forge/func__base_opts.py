import copy
import sys
from functools import wraps
from types import FunctionType
import param
from . import util
from .pprint import PrettyPrinter
def _base_opts(self, *args, **kwargs):
    from .options import Options
    new_args = []
    for arg in args:
        if isinstance(arg, Options) and arg.key is None:
            arg = arg(key=type(self._obj).__name__)
        new_args.append(arg)
    apply_groups, options, new_kwargs = util.deprecated_opts_signature(new_args, kwargs)
    clone = kwargs.get('clone', None)
    if apply_groups:
        from ..util import opts
        if options is not None:
            kwargs['options'] = options
        return opts.apply_groups(self._obj, **dict(kwargs, **new_kwargs))
    kwargs['clone'] = False if clone is None else clone
    return self._obj.options(*new_args, **kwargs)