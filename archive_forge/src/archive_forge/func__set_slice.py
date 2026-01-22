from functools import total_ordering
def _set_slice(self, index, values):
    """Assign values to a slice of the object"""
    try:
        valueList = list(values)
    except TypeError:
        raise TypeError('can only assign an iterable to a slice')
    self._check_allowed(valueList)
    origLen = len(self)
    start, stop, step = index.indices(origLen)
    if index.step is None:
        self._assign_simple_slice(start, stop, valueList)
    else:
        self._assign_extended_slice(start, stop, step, valueList)