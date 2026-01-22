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
class _ReferenceDict(MutableMapping):
    """A dict-like object whose values are defined by a slice.

    This implements a dict-like object whose keys and values are defined
    by a component slice (:py:class:`IndexedComponent_slice`).  The
    intent behind this object is to replace the normal ``_data``
    :py:class:`dict` in :py:class:`IndexedComponent` containers to
    create "reference" components.

    Parameters
    ----------
    component_slice : :py:class:`IndexedComponent_slice`
        The slice object that defines the "members" of this mutable mapping.
    """

    def __init__(self, component_slice):
        self._slice = component_slice

    def __contains__(self, key):
        try:
            next(self._get_iter(self._slice, key))
            return True
        except SliceEllipsisLookupError:
            if type(key) is tuple and len(key) == 1:
                key = key[0]
            _iter = iter(self._slice)
            for item in _iter:
                if _iter.get_last_index_wildcards() == key:
                    return True
            return False
        except StopIteration:
            return False
        except LookupError as e:
            if normalize_index.flatten:
                return False
            try:
                next(self._get_iter(self._slice, (key,)))
                return True
            except LookupError:
                return False

    def __getitem__(self, key):
        try:
            return next(self._get_iter(self._slice, key, get_if_not_present=True))
        except SliceEllipsisLookupError:
            if type(key) is tuple and len(key) == 1:
                key = key[0]
            _iter = iter(self._slice)
            for item in _iter:
                if _iter.get_last_index_wildcards() == key:
                    return item
            raise KeyError('KeyError: %s' % (key,))
        except (StopIteration, LookupError):
            raise KeyError('KeyError: %s' % (key,))

    def __setitem__(self, key, val):
        tmp = self._slice.duplicate()
        op = tmp._call_stack[-1][0]
        if op == IndexedComponent_slice.get_item:
            tmp._call_stack[-1] = (IndexedComponent_slice.set_item, tmp._call_stack[-1][1], val)
        elif op == IndexedComponent_slice.slice_info:
            tmp._call_stack[-1] = (IndexedComponent_slice.set_item, tmp._call_stack[-1][1], val)
        elif op == IndexedComponent_slice.get_attribute:
            tmp._call_stack[-1] = (IndexedComponent_slice.set_attribute, tmp._call_stack[-1][1], val)
        else:
            raise DeveloperError('Unexpected slice _call_stack operation: %s' % op)
        try:
            next(self._get_iter(tmp, key, get_if_not_present=True))
        except StopIteration:
            pass

    def __delitem__(self, key):
        tmp = self._slice.duplicate()
        op = tmp._call_stack[-1][0]
        if op == IndexedComponent_slice.get_item:
            tmp._call_stack[-1] = (IndexedComponent_slice.del_item, tmp._call_stack[-1][1])
        elif op == IndexedComponent_slice.slice_info:
            assert len(tmp._call_stack) == 1
            _iter = self._get_iter(tmp, key)
            next(_iter)
            del _iter._iter_stack[0].component[_iter.get_last_index()]
            return
        elif op == IndexedComponent_slice.get_attribute:
            tmp._call_stack[-1] = (IndexedComponent_slice.del_attribute, tmp._call_stack[-1][1])
        else:
            raise DeveloperError('Unexpected slice _call_stack operation: %s' % op)
        try:
            next(self._get_iter(tmp, key))
        except StopIteration:
            pass

    def __iter__(self):
        return self._slice.wildcard_keys(SortComponents.UNSORTED)

    def __len__(self):
        return sum((1 for i in self._slice))

    def keys(self, sort=SortComponents.UNSORTED):
        return self._slice.wildcard_keys(sort)

    def items(self, sort=SortComponents.UNSORTED):
        """Return the wildcard, value tuples for this ReferenceDict

        This method is necessary because the default implementation
        iterates over the keys and looks the values up in the
        dictionary.  Unfortunately some slices have structures that make
        looking up components by the wildcard keys very expensive
        (linear time; e.g., the use of ellipses with jagged sets).  By
        implementing this method without using lookups, general methods
        that iterate over everything (like component.pprint()) will
        still be linear and not quadratic time.

        """
        return self._slice.wildcard_items(sort)

    def values(self, sort=SortComponents.UNSORTED):
        """Return the values for this ReferenceDict

        This method is necessary because the default implementation
        iterates over the keys and looks the values up in the
        dictionary.  Unfortunately some slices have structures that make
        looking up components by the wildcard keys very expensive
        (linear time; e.g., the use of ellipses with jagged sets).  By
        implementing this method without using lookups, general methods
        that iterate over everything (like component.pprint()) will
        still be linear and not quadratic time.

        """
        return self._slice.wildcard_values(sort)

    @deprecated('The iteritems method is deprecated. Use dict.items().', version='6.0')
    def iteritems(self):
        return self.items()

    @deprecated('The itervalues method is deprecated. Use dict.values().', version='6.0')
    def itervalues(self):
        return self.values()

    def _get_iter(self, _slice, key, get_if_not_present=False):
        if key.__class__ not in (tuple, list):
            key = (key,)
        if normalize_index.flatten:
            key = flatten_tuple(key)
        return _IndexedComponent_slice_iter(_slice, _fill_in_known_wildcards(key, get_if_not_present=get_if_not_present))