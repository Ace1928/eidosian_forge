import copy
import sys
from functools import wraps
from types import FunctionType
import param
from . import util
from .pprint import PrettyPrinter
def _holomap_opts(self, *args, clone=None, **kwargs):
    apply_groups, _, _ = util.deprecated_opts_signature(args, kwargs)
    data = dict([(k, v.opts(*args, **kwargs)) for k, v in self._obj.data.items()])
    if apply_groups if clone is None else clone:
        return self._obj.clone(data)
    else:
        self._obj.data = data
        return self._obj