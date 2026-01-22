import itertools
import types
import unittest
from cupy.testing import _bundle
from cupy.testing import _pytest_impl
def _make_class_name(base_class_name, i_param, param):
    SINGLE_PARAM_MAXLEN = 100
    PARAMS_MAXLEN = 5000
    param_strs = ['{}={}'.format(k, _shorten(_param_to_str(v), SINGLE_PARAM_MAXLEN)) for k, v in sorted(param.items())]
    param_strs = _shorten(', '.join(param_strs), PARAMS_MAXLEN)
    cls_name = '{}_param_{}_{{{}}}'.format(base_class_name, i_param, param_strs)
    return cls_name