import copy
import os
import pickle
import warnings
import numpy as np
def _interpretIndex(self, ind, pos, numOk):
    if type(ind) is int:
        if not numOk:
            raise Exception('string and integer indexes may not follow named indexes')
        return (pos, ind, False)
    if MetaArray.isNameType(ind):
        if not numOk:
            raise Exception('string and integer indexes may not follow named indexes')
        return (pos, self._getIndex(pos, ind), False)
    elif type(ind) is slice:
        if MetaArray.isNameType(ind.start) or MetaArray.isNameType(ind.stop):
            axis = self._interpretAxis(ind.start)
            if MetaArray.isNameType(ind.stop):
                index = self._getIndex(axis, ind.stop)
            elif (isinstance(ind.stop, float) or isinstance(ind.step, float)) and 'values' in self._info[axis]:
                if ind.stop is None:
                    mask = self.xvals(axis) < ind.step
                elif ind.step is None:
                    mask = self.xvals(axis) >= ind.stop
                else:
                    mask = (self.xvals(axis) >= ind.stop) * (self.xvals(axis) < ind.step)
                index = mask
            elif isinstance(ind.stop, int) or isinstance(ind.step, int):
                if ind.step is None:
                    index = ind.stop
                else:
                    index = slice(ind.stop, ind.step)
            elif type(ind.stop) is list:
                index = []
                for i in ind.stop:
                    if type(i) is int:
                        index.append(i)
                    elif MetaArray.isNameType(i):
                        index.append(self._getIndex(axis, i))
                    else:
                        index = ind.stop
                        break
            else:
                index = ind.stop
            return (axis, index, True)
        else:
            return (pos, ind, False)
    elif type(ind) is list:
        indList = [self._interpretIndex(i, pos, numOk)[1] for i in ind]
        return (pos, indList, False)
    else:
        if not numOk:
            raise Exception('string and integer indexes may not follow named indexes')
        return (pos, ind, False)