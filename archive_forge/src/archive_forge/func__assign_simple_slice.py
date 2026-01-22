from functools import total_ordering
def _assign_simple_slice(self, start, stop, valueList):
    """Assign a simple slice; Can assign slice of any length"""
    origLen = len(self)
    stop = max(start, stop)
    newLen = origLen - stop + start + len(valueList)

    def newItems():
        for i in range(origLen + 1):
            if i == start:
                yield from valueList
            if i < origLen:
                if i < start or i >= stop:
                    yield self._get_single_internal(i)
    self._rebuild(newLen, newItems())