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
class _OrderedSetMixin(object):
    __slots__ = ()
    _valid_getitem_keys = {None, (None,), Ellipsis}

    def at(self, index):
        raise DeveloperError('Derived ordered set class (%s) failed to implement at' % (type(self).__name__,))

    def ord(self, val):
        raise DeveloperError('Derived ordered set class (%s) failed to implement ord' % (type(self).__name__,))

    def __getitem__(self, key):
        if not self.is_indexed() and (key in self._valid_getitem_keys or type(key) is slice):
            return super().__getitem__(key)
        deprecation_warning('Using __getitem__ to return a set value from its (ordered) position is deprecated.  Please use at()', version='6.1', remove_in='7.0')
        return self.at(key)

    @deprecated('card() was incorrectly added to the Set API.  Please use at()', version='6.1.2', remove_in='6.2')
    def card(self, index):
        return self.at(index)

    def isordered(self):
        """Returns True if this is an ordered finite discrete (iterable) Set"""
        return True

    def ordered_data(self):
        return self.data()

    def ordered_iter(self):
        return iter(self)

    def first(self):
        return self.at(1)

    def last(self):
        return self.at(len(self))

    def next(self, item, step=1):
        """
        Return the next item in the set.

        The default behavior is to return the very next element. The `step`
        option can specify how many steps are taken to get the next element.

        If the search item is not in the Set, or the next element is beyond
        the end of the set, then an IndexError is raised.
        """
        position = self.ord(item) + step
        if position < 1:
            raise IndexError('Cannot advance before the beginning of the Set')
        if position > len(self):
            raise IndexError('Cannot advance past the end of the Set')
        return self.at(position)

    def nextw(self, item, step=1):
        """
        Return the next item in the set with wrapping if necessary.

        The default behavior is to return the very next element. The `step`
        option can specify how many steps are taken to get the next element.
        If the next element is past the end of the Set, the search wraps back
        to the beginning of the Set.

        If the search item is not in the Set an IndexError is raised.
        """
        position = self.ord(item)
        return self.at((position + step - 1) % len(self) + 1)

    def prev(self, item, step=1):
        """Return the previous item in the set.

        The default behavior is to return the immediately previous
        element. The `step` option can specify how many steps are taken
        to get the previous element.

        If the search item is not in the Set, or the previous element is
        before the beginning of the set, then an IndexError is raised.
        """
        return self.next(item, -step)

    def prevw(self, item, step=1):
        """Return the previous item in the set with wrapping if necessary.

        The default behavior is to return the immediately
        previouselement. The `step` option can specify how many steps
        are taken to get the previous element. If the previous element
        is past the end of the Set, the search wraps back to the end of
        the Set.

        If the search item is not in the Set an IndexError is raised.
        """
        return self.nextw(item, -step)

    def _to_0_based_index(self, item):
        try:
            _item = int(item)
            if item != _item:
                raise IndexError()
        except:
            raise IndexError(f"Set '{self.name}' positional indices must be integers, not {type(item).__name__}") from None
        if _item >= 1:
            return _item - 1
        elif _item < 0:
            _item += len(self)
            if _item < 0:
                raise IndexError(f'{self.name} index out of range')
            return _item
        else:
            raise IndexError('Accessing Pyomo Sets by position is 1-based: valid Set positional index values are [1 .. len(Set)] or [-1 .. -len(Set)]')