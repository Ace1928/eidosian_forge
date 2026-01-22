import copy
import itertools
from pyomo.common import DeveloperError
from pyomo.common.collections import Sequence
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_index
class IndexedComponent_slice(object):
    """Special class for slicing through hierarchical component trees

    The basic concept is to interrupt the normal slice generation
    procedure to return a specialized iterable class (this object).  This
    object supports simple getitem / getattr / call methods and caches
    them until it is time to actually iterate through the slice.  We
    then walk down the cached names / indices and resolve the final
    objects during the iteration process.  This works because all the
    calls to __getitem__ / __getattr__ / __call__ happen *before* the
    call to __iter__()
    """
    ATTR_MASK = 8
    ITEM_MASK = 16
    CALL_MASK = 32
    GET_MASK = 1
    SET_MASK = 2
    DEL_MASK = 4
    slice_info = 0
    get_attribute = ATTR_MASK | GET_MASK
    set_attribute = ATTR_MASK | SET_MASK
    del_attribute = ATTR_MASK | DEL_MASK
    get_item = ITEM_MASK | GET_MASK
    set_item = ITEM_MASK | SET_MASK
    del_item = ITEM_MASK | DEL_MASK
    call = CALL_MASK

    def __init__(self, component, fixed=None, sliced=None, ellipsis=None):
        """A "slice" over an _IndexedComponent hierarchy

        This class has two forms for the constructor.  The first form is
        the standard constructor that takes a base component and
        indexing information.  This form takes

           IndexedComponent_slice(component, fixed, sliced, ellipsis)

        The second form is a "copy constructor" that is used internally
        when building up the "call stack" for the hierarchical slice.  The
        copy constructor takes an IndexedComponent_slice and an
        optional "next term" in the slice construction (from get/set/del
        item/attr or call):

           IndexedComponent_slice(slice, next_term=None)

        Parameters
        ----------
        component: IndexedComponent
            The base component for this slice

        fixed: dict
            A dictionary indicating the fixed indices of component,
            mapping index position to value

        sliced: dict
            A dictionary indicating the sliced indices of component
            mapping the index position to the (python) slice object

        ellipsis: int
            The position of the ellipsis in the initial component slice

        """
        set_attr = super(IndexedComponent_slice, self).__setattr__
        if type(component) is IndexedComponent_slice:
            _len = component._len
            if _len == len(component._call_stack):
                set_attr('_call_stack', component._call_stack)
            else:
                set_attr('_call_stack', component._call_stack[:_len])
            set_attr('_len', _len)
            if fixed is not None:
                self._call_stack.append(fixed)
                self._len += 1
            set_attr('call_errors_generate_exceptions', component.call_errors_generate_exceptions)
            set_attr('key_errors_generate_exceptions', component.key_errors_generate_exceptions)
            set_attr('attribute_errors_generate_exceptions', component.attribute_errors_generate_exceptions)
        else:
            set_attr('_call_stack', [(IndexedComponent_slice.slice_info, (component, fixed, sliced, ellipsis))])
            set_attr('_len', 1)
            set_attr('call_errors_generate_exceptions', True)
            set_attr('key_errors_generate_exceptions', True)
            set_attr('attribute_errors_generate_exceptions', True)

    def __getstate__(self):
        """Serialize this object.

        In general, we would not need to implement this (the object does
        not leverage ``__slots__``).  However, because we have a
        "blanket" implementation of :py:meth:`__getattr__`, we need to
        explicitly implement these to avoid "accidentally" extending or
        evaluating this slice."""
        return {k: getattr(self, k) for k in self.__dict__}

    def __setstate__(self, state):
        """Deserialize the state into this object."""
        set_attr = super(IndexedComponent_slice, self).__setattr__
        for k, v in state.items():
            set_attr(k, v)

    def __deepcopy__(self, memo):
        """Deepcopy this object (leveraging :py:meth:`__getstate__`)"""
        ans = memo[id(self)] = self.__class__.__new__(self.__class__)
        ans.__setstate__(copy.deepcopy(self.__getstate__(), memo))
        return ans

    def __iter__(self):
        """Return an iterator over this slice"""
        return _IndexedComponent_slice_iter(self)

    def __getattr__(self, name):
        """Override the "." operator to defer resolution until iteration.

        Creating a slice of a component returns a
        IndexedComponent_slice object.  Subsequent attempts to resolve
        attributes hit this method.
        """
        return IndexedComponent_slice(self, (IndexedComponent_slice.get_attribute, name))

    def __setattr__(self, name, value):
        """Override the "." operator implementing attribute assignment

        This supports notation similar to:

            del model.b[:].c.x = 5

        and immediately evaluates the slice.
        """
        if name in self.__dict__:
            return super(IndexedComponent_slice, self).__setattr__(name, value)
        for i in IndexedComponent_slice(self, (IndexedComponent_slice.set_attribute, name, value)):
            pass
        return None

    def __getitem__(self, idx):
        """Override the "[]" operator to defer resolution until iteration.

        Creating a slice of a component returns a
        IndexedComponent_slice object.  Subsequent attempts to query
        items hit this method.
        """
        return IndexedComponent_slice(self, (IndexedComponent_slice.get_item, idx))

    def __setitem__(self, idx, val):
        """Override the "[]" operator for setting item values.

        This supports notation similar to:

            model.b[:].c.x[1,:] = 5

        and immediately evaluates the slice.
        """
        for i in IndexedComponent_slice(self, (IndexedComponent_slice.set_item, idx, val)):
            pass
        return None

    def __delitem__(self, idx):
        """Override the "del []" operator for deleting item values.

        This supports notation similar to:

            del model.b[:].c.x[1,:]

        and immediately evaluates the slice.
        """
        for i in IndexedComponent_slice(self, (IndexedComponent_slice.del_item, idx)):
            pass
        return None

    def __call__(self, *args, **kwds):
        """Special handling of the "()" operator for component slices.

        Creating a slice of a component returns a IndexedComponent_slice
        object.  Subsequent attempts to call items hit this method.  We
        handle the __call__ method separately based on the item (identifier
        immediately before the "()") being called:

        - if the item was 'component', then we defer resolution of this call
        until we are actually iterating over the slice.  This allows users
        to do operations like `m.b[:].component('foo').bar[:]`

        - if the item is anything else, then we will immediately iterate over
        the slice and call the item.  This allows "vector-like" operations
        like: `m.x[:,1].fix(0)`.
        """
        _len = self._len
        if self._call_stack[_len - 1][0] == IndexedComponent_slice.get_attribute and self._call_stack[_len - 1][1] == '__name__':
            self._len -= 1
        ans = IndexedComponent_slice(self, (IndexedComponent_slice.call, args, kwds))
        if ans._call_stack[-2][1] == 'component':
            return ans
        else:
            return list((i for i in ans))

    @classmethod
    def _getitem_args_to_str(cls, args):
        for i, v in enumerate(args):
            if v is Ellipsis:
                args[i] = '...'
            elif type(v) is slice:
                args[i] = (repr(v.start) if v.start is not None else '') + ':' + (repr(v.stop) if v.stop is not None else '') + (':%r' % v.step if v.step is not None else '')
            else:
                args[i] = repr(v)
        return '[' + ', '.join(args) + ']'

    def __str__(self):
        ans = ''
        for level in self._call_stack:
            if level[0] == IndexedComponent_slice.slice_info:
                ans += level[1][0].name
                if level[1][1] is not None:
                    tmp = dict(level[1][1])
                    tmp.update(level[1][2])
                    if level[1][3] is not None:
                        tmp[level[1][3]] = Ellipsis
                    ans += self._getitem_args_to_str([tmp[i] for i in sorted(tmp)])
            elif level[0] & IndexedComponent_slice.ITEM_MASK:
                if isinstance(level[1], Sequence):
                    tmp = list(level[1])
                else:
                    tmp = [level[1]]
                ans += self._getitem_args_to_str(tmp)
            elif level[0] & IndexedComponent_slice.ATTR_MASK:
                ans += '.' + level[1]
            elif level[0] & IndexedComponent_slice.CALL_MASK:
                ans += '(' + ', '.join(itertools.chain((repr(_) for _ in level[1]), ('%s=%r' % kv for kv in level[2].items()))) + ')'
            if level[0] & IndexedComponent_slice.SET_MASK:
                ans += ' = %r' % (level[2],)
            elif level[0] & IndexedComponent_slice.DEL_MASK:
                ans = 'del ' + ans
        return ans

    def __hash__(self):
        return hash(tuple((_freeze(x) for x in self._call_stack[:self._len])))

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is not IndexedComponent_slice:
            return False
        return tuple((_freeze(x) for x in self._call_stack[:self._len])) == tuple((_freeze(x) for x in other._call_stack[:other._len]))

    def __ne__(self, other):
        return not self.__eq__(other)

    def duplicate(self):
        ans = IndexedComponent_slice(self)
        ans._call_stack = ans._call_stack[:ans._len]
        return ans

    def index_wildcard_keys(self, sort):
        _iter = _IndexedComponent_slice_iter(self, iter_over_index=True, sort=sort)
        return (_iter.get_last_index_wildcards() for _ in _iter)

    def wildcard_keys(self, sort=SortComponents.UNSORTED):
        _iter = _IndexedComponent_slice_iter(self, sort=sort)
        return (_iter.get_last_index_wildcards() for _ in _iter)

    def wildcard_values(self, sort=SortComponents.UNSORTED):
        """Return an iterator over this slice"""
        return _IndexedComponent_slice_iter(self, sort=sort)

    def wildcard_items(self, sort=SortComponents.UNSORTED):
        _iter = _IndexedComponent_slice_iter(self, sort=sort)
        return ((_iter.get_last_index_wildcards(), _) for _ in _iter)

    def expanded_keys(self):
        _iter = self.__iter__()
        return (_iter.get_last_index() for _ in _iter)

    def expanded_items(self):
        _iter = self.__iter__()
        return ((_iter.get_last_index(), _) for _ in _iter)