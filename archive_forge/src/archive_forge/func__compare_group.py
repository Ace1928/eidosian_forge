import sys
from Cython.Tempita.compat3 import basestring_
def _compare_group(self, item, other, getter):
    if getter is None:
        return item != other
    elif isinstance(getter, basestring_) and getter.startswith('.'):
        getter = getter[1:]
        if getter.endswith('()'):
            getter = getter[:-2]
            return getattr(item, getter)() != getattr(other, getter)()
        else:
            return getattr(item, getter) != getattr(other, getter)
    elif hasattr(getter, '__call__'):
        return getter(item) != getter(other)
    else:
        return item[getter] != other[getter]