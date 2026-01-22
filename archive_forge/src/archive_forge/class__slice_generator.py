import copy
import itertools
from pyomo.common import DeveloperError
from pyomo.common.collections import Sequence
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_index
class _slice_generator(object):
    """Utility (iterator) for generating the elements of one slice

    Iterate through the component index and yield the component data
    values that match the slice template.
    """

    def __init__(self, component, fixed, sliced, ellipsis, iter_over_index, sort):
        self.component = component
        self.fixed = fixed
        self.sliced = sliced
        self.ellipsis = ellipsis
        self.iter_over_index = iter_over_index
        self.last_index = ()
        self.tuplize_unflattened_index = len(list(self.component.index_set().subsets())) <= 1
        if fixed is None and sliced is None and (ellipsis is None):
            self.explicit_index_count = 0
            self.component_iter = _NotIterable
            return
        self.explicit_index_count = len(fixed) + len(sliced)
        if iter_over_index and component.index_set().isfinite():
            if SortComponents.SORTED_INDICES in sort:
                self.component_iter = component.index_set().sorted_iter()
            elif SortComponents.ORDERED_INDICES in sort:
                self.component_iter = component.index_set().ordered_iter()
            else:
                self.component_iter = iter(component.index_set())
        else:
            self.component_iter = component.keys(sort)

    def next(self):
        """__next__() iterator for Py2 compatibility"""
        return self.__next__()

    def __next__(self):
        from .indexed_component import normalize_index
        if self.component_iter is _NotIterable:
            self.component_iter = iter(())
            return self.component
        while 1:
            index = next(self.component_iter)
            if normalize_index.flatten:
                _idx = index if type(index) is tuple else (index,)
            elif self.tuplize_unflattened_index:
                _idx = (index,)
            else:
                _idx = index
            if self.ellipsis is not None:
                if self.explicit_index_count > len(_idx):
                    continue
            elif len(_idx) != self.explicit_index_count:
                continue
            valid = True
            for key, val in self.fixed.items():
                if not val == _idx[key]:
                    valid = False
                    break
            if valid:
                self.last_index = _idx
                if not self.iter_over_index or index in self.component:
                    return self.component[index]
                else:
                    return None