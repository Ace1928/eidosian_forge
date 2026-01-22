from pyomo.common import DeveloperError
from pyomo.common.collections import (
from pyomo.common.modeling import NOTSET
from pyomo.core.base.set import DeclareGlobalSet, Set, SetOf, OrderedSetOf, _SetDataBase
from pyomo.core.base.component import Component, ComponentData
from pyomo.core.base.global_set import UnindexedComponent_set
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.indexed_component import IndexedComponent, normalize_index
from pyomo.core.base.indexed_component_slice import (
from pyomo.core.base.util import flatten_tuple
from pyomo.common.deprecation import deprecated
class _fill_in_known_wildcards(object):
    """Variant of "six.advance_iterator" that substitutes wildcard values

    This object is initialized with a tuple of index values.  Calling
    the resulting object on a :py:class:`_slice_generator` will
    "advance" the iterator, substituting values from the tuple into the
    slice wildcards (":" indices), and returning the resulting object.
    The motivation for implementing this as an iterator is so that we
    can re-use all the logic from
    :py:meth:`_IndexedComponent_slice_iter.__next__` when looking up
    specific indices within the slice.

    Parameters
    ----------
    wildcard_values : tuple of index values
        a tuple containing index values to substitute into the slice wildcards

    look_in_index : :py:class:`bool` [optional]
        If True, the iterator will also look for matches using the
        components' underlying index_set() in addition to the (sparse)
        indices matched by the components' __contains__()
        method. [default: False]

    get_if_not_present : :py:class:`bool` [optional]
        If True, the iterator will attempt to retrieve data objects
        (through getitem) for indexes that match the underlying
        component index_set() but do not appear in the (sparse) indices
        matched by __contains__.  get_if_not_present implies
        look_in_index.  [default: False]

    """

    def __init__(self, wildcard_values, look_in_index=False, get_if_not_present=False):
        self.base_key = wildcard_values
        self.key = list(wildcard_values)
        self.known_slices = set()
        self.look_in_index = look_in_index or get_if_not_present
        self.get_if_not_present = get_if_not_present

    def __call__(self, _slice):
        """Advance the specified slice generator, substituting wildcard values

        This advances the passed :py:class:`_slice_generator
        <pyomo.core.base.indexed_component_slice._slice_generator>` by
        substituting values from the `wildcard_values` list for any
        wildcard slices ("`:`").

        Parameters
        ----------
        _slice : pyomo.core.base.indexed_component_slice._slice_generator
            the slice to advance
        """
        if _slice in self.known_slices:
            raise StopIteration()
        self.known_slices.add(_slice)
        if _slice.ellipsis is None:
            idx_count = _slice.explicit_index_count
        elif not _slice.component.is_indexed():
            idx_count = 1
        else:
            idx_count = _slice.component.index_set().dimen
            if idx_count is None:
                raise SliceEllipsisLookupError('Cannot lookup elements in a _ReferenceDict when the underlying slice object contains ellipsis over a jagged (dimen=None) Set')
        try:
            idx = tuple((_slice.fixed[i] if i in _slice.fixed else self.key.pop(0) for i in range(idx_count)))
        except IndexError:
            raise KeyError("Insufficient values for slice of indexed component '%s' (found evaluating slice index %s)" % (_slice.component.name, self.base_key))
        if idx in _slice.component:
            _slice.last_index = idx
            return _slice.component[idx]
        elif len(idx) == 1 and idx[0] in _slice.component:
            _slice.last_index = idx
            return _slice.component[idx[0]]
        elif not idx:
            return _slice.component
        elif self.look_in_index:
            if idx in _slice.component.index_set():
                _slice.last_index = idx
                return _slice.component[idx] if self.get_if_not_present else None
            elif len(idx) == 1 and idx[0] in _slice.component.index_set():
                _slice.last_index = idx
                return _slice.component[idx[0]] if self.get_if_not_present else None
        raise KeyError("Index %s is not valid for indexed component '%s' (found evaluating slice index %s)" % (idx, _slice.component.name, self.base_key))

    def check_complete(self):
        if not self.key:
            return
        if self.key == _UnindexedComponent_key and self.base_key == _UnindexedComponent_base_key:
            return
        raise KeyError('Extra (unused) values for slice index %s' % (self.base_key,))