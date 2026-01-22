import sys
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER
import locale
from _pydev_bundle import pydev_log
def _repr_dict(self, obj, level, prefix, suffix, item_prefix, item_sep, item_suffix):
    if not obj:
        yield (prefix + suffix)
        return
    if level >= len(self.maxcollection):
        yield (prefix + '...' + suffix)
        return
    yield prefix
    count = self.maxcollection[level]
    yield_comma = False
    if IS_PY36_OR_GREATER:
        sorted_keys = list(obj)
    else:
        try:
            sorted_keys = sorted(obj)
        except Exception:
            sorted_keys = list(obj)
    for key in sorted_keys:
        if yield_comma:
            yield ', '
        yield_comma = True
        count -= 1
        if count <= 0:
            yield '...'
            break
        yield item_prefix
        for p in self._repr(key, level + 1):
            yield p
        yield item_sep
        try:
            item = obj[key]
        except Exception:
            yield '<?>'
        else:
            for p in self._repr(item, 100 if item is obj else level + 1):
                yield p
        yield item_suffix
    yield suffix