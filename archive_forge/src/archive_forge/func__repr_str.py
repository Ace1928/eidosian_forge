import sys
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER
import locale
from _pydev_bundle import pydev_log
def _repr_str(self, obj, level):
    try:
        if self.raw_value:
            if isinstance(obj, bytes):
                yield obj.decode('latin-1')
            else:
                yield obj
            return
        limit_inner = self.maxother_inner
        limit_outer = self.maxother_outer
        limit = limit_inner if level > 0 else limit_outer
        if len(obj) <= limit:
            yield self._convert_to_unicode_or_bytes_repr(repr(obj))
            return
        left_count, right_count = (max(1, int(2 * limit / 3)), max(1, int(limit / 3)))
        part1 = obj[:left_count]
        part1 = repr(part1)
        part1 = part1[:part1.rindex("'")]
        part2 = obj[-right_count:]
        part2 = repr(part2)
        part2 = part2[part2.index("'") + 1:]
        yield part1
        yield '...'
        yield part2
    except:
        pydev_log.exception('Error getting string representation to show.')
        for part in self._repr_obj(obj, level, self.maxother_inner, self.maxother_outer):
            yield part