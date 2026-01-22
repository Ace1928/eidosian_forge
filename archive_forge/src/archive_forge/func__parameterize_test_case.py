import itertools
import types
import unittest
from cupy.testing import _bundle
from cupy.testing import _pytest_impl
def _parameterize_test_case(base, i, param):
    cls_name = _make_class_name(base.__name__, i, param)

    def __repr__(self):
        name = base.__repr__(self)
        return '<%s  parameter: %s>' % (name, param)
    mb = {'__repr__': __repr__}
    for k, v in sorted(param.items()):
        if isinstance(v, types.FunctionType):

            def create_new_v():
                f = v

                def new_v(self, *args, **kwargs):
                    return f(*args, **kwargs)
                return new_v
            mb[k] = create_new_v()
        else:
            mb[k] = v
    return (cls_name, mb, lambda method: method)