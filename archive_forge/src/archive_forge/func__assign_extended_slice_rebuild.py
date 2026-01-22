from functools import total_ordering
def _assign_extended_slice_rebuild(self, start, stop, step, valueList):
    """Assign an extended slice by rebuilding entire list"""
    indexList = range(start, stop, step)
    if len(valueList) != len(indexList):
        raise ValueError('attempt to assign sequence of size %d to extended slice of size %d' % (len(valueList), len(indexList)))
    newLen = len(self)
    newVals = dict(zip(indexList, valueList))

    def newItems():
        for i in range(newLen):
            if i in newVals:
                yield newVals[i]
            else:
                yield self._get_single_internal(i)
    self._rebuild(newLen, newItems())