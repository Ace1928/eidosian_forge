import sys
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER
import locale
from _pydev_bundle import pydev_log
def _repr_iter(self, obj, level, prefix, suffix, comma_after_single_element=False):
    yield prefix
    if level >= len(self.maxcollection):
        yield '...'
    else:
        count = self.maxcollection[level]
        yield_comma = False
        for item in obj:
            if yield_comma:
                yield ', '
            yield_comma = True
            count -= 1
            if count <= 0:
                yield '...'
                break
            for p in self._repr(item, 100 if item is obj else level + 1):
                yield p
        else:
            if comma_after_single_element:
                if count == self.maxcollection[level] - 1:
                    yield ','
    yield suffix