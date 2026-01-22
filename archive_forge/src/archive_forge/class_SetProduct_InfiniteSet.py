import inspect
import itertools
import logging
import math
import sys
import weakref
from pyomo.common.pyomo_typing import overload
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.errors import DeveloperError, PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
from pyomo.core.base.range import (
from pyomo.core.base.component import (
from pyomo.core.base.indexed_component import (
from pyomo.core.base.global_set import (
from collections.abc import Sequence
from operator import itemgetter
class SetProduct_InfiniteSet(SetProduct):
    __slots__ = tuple()

    def get(self, val, default=None):
        v = self._find_val(val)
        if v is None:
            return default
        if normalize_index.flatten:
            return self._flatten_product(v[0])
        return v[0]

    def _find_val(self, val):
        """Locate a value in this SetProduct

        Locate a value in this SetProduct.  Returns None if the value is
        not found, otherwise returns a (value, cutpoints) tuple.  Value
        is the value that was searched for, possibly normalized.
        Cutpoints is the set of indices that specify how to split the
        value into the corresponding subsets such that subset[i] =
        cutpoints[i:i+1].  Cutpoints is None if the value is trivially
        split with a single index for each subset.

        Returns
        -------
        val: tuple
        cutpoints: list
        """
        if hasattr(val, '__len__') and len(val) == len(self._sets):
            if all((v in self._sets[i] for i, v in enumerate(val))):
                return (val, None)
        if not normalize_index.flatten:
            return None
        val = normalize_index(val)
        if val.__class__ is tuple:
            v_len = len(val)
        else:
            val = (val,)
            v_len = 1
        setDims = list((s.dimen for s in self._sets))
        for i, d in enumerate(setDims):
            if d is UnknownSetDimen:
                setDims[i] = None
        index = [None] * len(setDims)
        lastIndex = 0
        for i, dim in enumerate(setDims):
            index[i] = lastIndex
            if dim is None:
                firstNonDimSet = i
                break
            lastIndex += dim
            if lastIndex > v_len:
                return None
            elif val[index[i]:lastIndex] not in self._sets[i]:
                return None
        index.append(v_len)
        if None not in setDims:
            if lastIndex == v_len:
                return (val, index)
            else:
                return None
        lastIndex = index[-1]
        for iEnd, dim in enumerate(reversed(setDims)):
            i = len(setDims) - (iEnd + 1)
            if dim is None:
                lastNonDimSet = i
                break
            lastIndex -= dim
            index[i] = lastIndex
            if val[index[i]:index[i + 1]] not in self._sets[i]:
                return None
        if firstNonDimSet == lastNonDimSet:
            if val[index[firstNonDimSet]:index[firstNonDimSet + 1]] in self._sets[firstNonDimSet]:
                return (val, index)
            else:
                return None
        subsets = self._sets[firstNonDimSet:lastNonDimSet + 1]
        _val = val[index[firstNonDimSet]:index[lastNonDimSet + 1]]
        for cuts in self._cutPointGenerator(subsets, len(_val)):
            if all((_val[cuts[i]:cuts[i + 1]] in s for i, s in enumerate(subsets))):
                offset = index[firstNonDimSet]
                for i in range(1, len(subsets)):
                    index[firstNonDimSet + i] = offset + cuts[i]
                return (val, index)
        return None

    @staticmethod
    def _cutPointGenerator(subsets, val_len):
        """Generate the sequence of cut points for a series of subsets.

        This generator produces the valid set of cut points for
        separating a list of length val_len into chunks that are valid
        for the specified subsets.  In this method, the first and last
        subsets must have dimen==None.  The return value is a list with
        length one greater that then number of subsets.  Value slices
        (for membership tests) are determined by

            cuts[i]:cuts[i+1] in subsets[i]

        """
        setDims = list((_.dimen for _ in subsets))
        cutIters = [None] * (len(subsets) + 1)
        cutPoints = [0] * (len(subsets) + 1)
        i = 1
        cutIters[i] = iter(range(val_len + 1))
        cutPoints[-1] = val_len
        while i > 0:
            try:
                cutPoints[i] = next(cutIters[i])
                if i < len(subsets) - 1:
                    if setDims[i] is not None:
                        cutIters[i + 1] = iter((cutPoints[i] + setDims[i],))
                    else:
                        cutIters[i + 1] = iter(range(cutPoints[i], val_len + 1))
                    i += 1
                elif cutPoints[i] > val_len:
                    i -= 1
                else:
                    yield cutPoints
            except StopIteration:
                i -= 1