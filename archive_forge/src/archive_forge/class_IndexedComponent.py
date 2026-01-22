import inspect
import logging
import sys
import textwrap
import pyomo.core.expr as EXPR
import pyomo.core.base as BASE
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.component import Component, ActiveComponent
from pyomo.core.base.config import PyomoOptions
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_set
from pyomo.core.expr.numeric_expr import _ndarray
from pyomo.core.pyomoobject import PyomoObject
from pyomo.common import DeveloperError
from pyomo.common.autoslots import fast_deepcopy
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import native_types
from pyomo.common.sorting import sorted_robust
from collections.abc import Sequence
class IndexedComponent(Component):
    """This is the base class for all indexed modeling components.
    This class stores a dictionary, self._data, that maps indices
    to component data objects.  The object self._index_set defines valid
    keys for this dictionary, and the dictionary keys may be a
    strict subset.

    The standard access and iteration methods iterate over the the
    keys of self._data.  This class supports a concept of a default
    component data value.  When enabled, the default does not
    change the access and iteration methods.

    IndexedComponent may be given a set over which indexing is restricted.
    Alternatively, IndexedComponent may be indexed over Any
    (pyomo.core.base.set_types.Any), in which case the IndexedComponent
    behaves like a dictionary - any hashable object can be used as a key
    and keys can be added on the fly.

    Constructor arguments:
        ctype       The class type for the derived subclass
        doc         A text string describing this component

    Private class attributes:

        _data:  A dictionary from the index set to component data objects

        _index_set:  The set of valid indices

        _anonymous_sets: A ComponentSet of "anonymous" sets used by this
            component.  Anonymous sets are Set / SetOperator / RangeSet
            that compose attributes like _index_set, but are not
            themselves explicitly assigned (and named) on any Block

    """

    class Skip(object):
        pass
    _DEFAULT_INDEX_CHECKING_ENABLED = True

    def __init__(self, *args, **kwds):
        kwds.pop('noruleinit', None)
        Component.__init__(self, **kwds)
        self._data = {}
        if len(args) == 0 or (args[0] is UnindexedComponent_set and len(args) == 1):
            self._index_set = UnindexedComponent_set
            self._anonymous_sets = None
        elif len(args) == 1:
            self._index_set, self._anonymous_sets = BASE.set.process_setarg(args[0])
        else:
            self._index_set = BASE.set.SetProduct(*args)
            self._anonymous_sets = ComponentSet((self._index_set,))
            if self._index_set._anonymous_sets is not None:
                self._anonymous_sets.update(self._index_set._anonymous_sets)

    def _create_objects_for_deepcopy(self, memo, component_list):
        _new = self.__class__.__new__(self.__class__)
        _ans = memo.setdefault(id(self), _new)
        if _ans is _new:
            component_list.append(self)
            if self.is_indexed() and (not self.is_reference()):
                _src = self._data
                memo[id(_src)] = _new._data = _data = _src.__class__()
                for idx, obj in _src.items():
                    _data[fast_deepcopy(idx, memo)] = obj._create_objects_for_deepcopy(memo, component_list)
        return _ans

    def to_dense_data(self):
        """TODO"""
        for idx in self._index_set:
            if idx in self._data:
                continue
            try:
                self._getitem_when_not_present(idx)
            except KeyError:
                pass

    def clear(self):
        """Clear the data in this component"""
        if self.is_indexed():
            self._data = {}
        else:
            raise DeveloperError('Derived scalar component %s failed to define clear().' % (self.__class__.__name__,))

    def index_set(self):
        """Return the index set"""
        return self._index_set

    def is_indexed(self):
        """Return true if this component is indexed"""
        return self._index_set is not UnindexedComponent_set

    def is_reference(self):
        """Return True if this component is a reference, where
        "reference" is interpreted as any component that does not
        own its own data.
        """
        return self._data is not None and type(self._data) is not dict

    def dim(self):
        """Return the dimension of the index"""
        if not self.is_indexed():
            return 0
        return self._index_set.dimen

    def __len__(self):
        """
        Return the number of component data objects stored by this
        component.
        """
        return len(self._data)

    def __contains__(self, idx):
        """Return true if the index is in the dictionary"""
        return idx in self._data

    def __iter__(self):
        """Return an iterator of the component data keys"""
        return self.keys()

    def keys(self, sort=SortComponents.UNSORTED, ordered=NOTSET):
        """Return an iterator over the component data keys

        This method sets the ordering of component data objects within
        this IndexedComponent container.  For consistency,
        :py:meth:`__init__()`, :py:meth:`values`, and :py:meth:`items`
        all leverage this method to ensure consistent ordering.

        Parameters
        ----------
        sort: bool or SortComponents
            Iterate over the declared component keys in a specified
            sorted order.  See :py:class:`SortComponents` for valid
            options and descriptions.

        ordered: bool
            DEPRECATED: Please use `sort=SortComponents.ORDERED_INDICES`.
            If True, then the keys are returned in a deterministic order
            (using the underlying set's `ordered_iter()`).

        """
        sort = SortComponents(sort)
        if ordered is not NOTSET:
            deprecation_warning(f'keys(ordered={ordered}) is deprecated.  Please use `sort=SortComponents.ORDERED_INDICES`', version='6.6.0')
            if ordered:
                sort = sort | SortComponents.ORDERED_INDICES
        if not self._index_set.isfinite():
            if SortComponents.SORTED_INDICES in sort or SortComponents.ORDERED_INDICES in sort:
                return iter(sorted_robust(self._data))
            else:
                return self._data.__iter__()
        if SortComponents.SORTED_INDICES in sort:
            ans = self._index_set.sorted_iter()
        elif SortComponents.ORDERED_INDICES in sort:
            ans = self._index_set.ordered_iter()
        else:
            ans = iter(self._index_set)
        if self._data.__class__ is not dict:
            pass
        elif len(self) == len(self._index_set):
            pass
        elif not self._data and self._index_set and PyomoOptions.paranoia_level:
            logger.warning("Iterating over a Component (%s)\ndefined by a non-empty concrete set before any data objects have\nactually been added to the Component.  The iterator will be empty.\nThis is usually caused by Concrete models where you declare the\ncomponent (e.g., a Var) and apply component-level operations (e.g.,\nx.fix(0)) before you use the component members (in something like a\nconstraint).\n\nYou can silence this warning by one of three ways:\n    1) Declare the component to be dense with the 'dense=True' option.\n       This will cause all data objects to be immediately created and\n       added to the Component.\n    2) Defer component-level iteration until after the component data\n       members have been added (through explicit use).\n    3) If you intend to iterate over a component that may be empty, test\n       if the component is empty first and avoid iteration in the case\n       where it is empty.\n" % (self.name,))
        else:
            ans = filter(self._data.__contains__, ans)
        return ans

    def values(self, sort=SortComponents.UNSORTED, ordered=NOTSET):
        """Return an iterator of the component data objects

        Parameters
        ----------
        sort: bool or SortComponents
            Iterate over the declared component values in a specified
            sorted order.  See :py:class:`SortComponents` for valid
            options and descriptions.

        ordered: bool
            DEPRECATED: Please use `sort=SortComponents.ORDERED_INDICES`.
            If True, then the values are returned in a deterministic order
            (using the underlying set's `ordered_iter()`.
        """
        if ordered is not NOTSET:
            deprecation_warning(f'values(ordered={ordered}) is deprecated.  Please use `sort=SortComponents.ORDERED_INDICES`', version='6.6.0')
            if ordered:
                sort = SortComponents(sort) | SortComponents.ORDERED_INDICES
        if self.is_reference():
            try:
                return self._data.values(sort)
            except TypeError:
                pass
        return map(self.__getitem__, self.keys(sort))

    def items(self, sort=SortComponents.UNSORTED, ordered=NOTSET):
        """Return an iterator of (index,data) component data tuples

        Parameters
        ----------
        sort: bool or SortComponents
            Iterate over the declared component items in a specified
            sorted order.  See :py:class:`SortComponents` for valid
            options and descriptions.

        ordered: bool
            DEPRECATED: Please use `sort=SortComponents.ORDERED_INDICES`.
            If True, then the items are returned in a deterministic order
            (using the underlying set's `ordered_iter()`.
        """
        if ordered is not NOTSET:
            deprecation_warning(f'items(ordered={ordered}) is deprecated.  Please use `sort=SortComponents.ORDERED_INDICES`', version='6.6.0')
            if ordered:
                sort = SortComponents(sort) | SortComponents.ORDERED_INDICES
        if self.is_reference():
            try:
                return self._data.items(sort)
            except TypeError:
                pass
        return ((s, self[s]) for s in self.keys(sort))

    @deprecated('The iterkeys method is deprecated. Use dict.keys().', version='6.0')
    def iterkeys(self):
        """Return a list of keys in the dictionary"""
        return self.keys()

    @deprecated('The itervalues method is deprecated. Use dict.values().', version='6.0')
    def itervalues(self):
        """Return a list of the component data objects in the dictionary"""
        return self.values()

    @deprecated('The iteritems method is deprecated. Use dict.items().', version='6.0')
    def iteritems(self):
        """Return a list (index,data) tuples from the dictionary"""
        return self.items()

    def __getitem__(self, index):
        """
        This method returns the data corresponding to the given index.
        """
        if self._constructed is False:
            self._not_constructed_error(index)
        try:
            return self._data[index]
        except KeyError:
            obj = _NotFound
        except TypeError:
            try:
                index = self._processUnhashableIndex(index)
            except TypeError:
                index = TypeError
            if index is TypeError:
                raise
            if index.__class__ is IndexedComponent_slice:
                return index
            try:
                obj = self._data.get(index, _NotFound)
            except TypeError:
                obj = _NotFound
        if obj is _NotFound:
            if isinstance(index, EXPR.GetItemExpression):
                return index
            validated_index = self._validate_index(index)
            if validated_index is not index:
                index = validated_index
                if index.__class__ is IndexedComponent_slice:
                    return index
                obj = self._data.get(index, _NotFound)
            if obj is _NotFound:
                return self._getitem_when_not_present(index)
        return obj

    def __setitem__(self, index, val):
        if self._constructed is False:
            self._not_constructed_error(index)
        try:
            obj = self._data.get(index, _NotFound)
        except TypeError:
            obj = _NotFound
            index = self._processUnhashableIndex(index)
        if obj is _NotFound:
            if index.__class__ is not IndexedComponent_slice:
                index = self._validate_index(index)
        else:
            return self._setitem_impl(index, obj, val)
        if index.__class__ is IndexedComponent_slice:
            assert len(index._call_stack) == 1
            for idx, obj in list(index.expanded_items()):
                self._setitem_impl(idx, obj, val)
        else:
            obj = self._data.get(index, _NotFound)
            if obj is _NotFound:
                return self._setitem_when_not_present(index, val)
            else:
                return self._setitem_impl(index, obj, val)

    def __delitem__(self, index):
        if self._constructed is False:
            self._not_constructed_error(index)
        try:
            obj = self._data.get(index, _NotFound)
        except TypeError:
            obj = _NotFound
            index = self._processUnhashableIndex(index)
        if obj is _NotFound:
            if index.__class__ is not IndexedComponent_slice:
                index = self._validate_index(index)
        if index.__class__ is IndexedComponent_slice:
            assert len(index._call_stack) == 1
            for idx in list(index.expanded_keys()):
                del self[idx]
        else:
            if self.is_indexed():
                self._data[index]._component = None
            del self._data[index]

    def _construct_from_rule_using_setitem(self):
        if self._rule is None:
            return
        index = None
        rule = self._rule
        block = self.parent_block()
        try:
            if rule.constant() and self.is_indexed():
                self._rule = rule = Initializer(rule(block, None), treat_sequences_as_mappings=False, arg_not_specified=NOTSET)
            if rule.contains_indices():
                for index in rule.indices():
                    self[index] = rule(block, index)
            elif not self.index_set().isfinite():
                pass
            elif rule.constant():
                val = rule(block, None)
                for index in self.index_set():
                    self._setitem_when_not_present(index, val)
            else:
                for index in self.index_set():
                    self._setitem_when_not_present(index, rule(block, index))
        except:
            err = sys.exc_info()[1]
            logger.error("Rule failed for %s '%s' with index %s:\n%s: %s" % (self.ctype.__name__, self.name, str(index), type(err).__name__, err))
            raise

    def _not_constructed_error(self, idx):
        if not self.is_indexed():
            idx_str = ''
        elif idx.__class__ is tuple:
            idx_str = '[' + ','.join((str(i) for i in idx)) + ']'
        else:
            idx_str = '[' + str(idx) + ']'
        raise ValueError('Error retrieving component %s%s: The component has not been constructed.' % (self.name, idx_str))

    def _validate_index(self, idx):
        if not IndexedComponent._DEFAULT_INDEX_CHECKING_ENABLED:
            return idx
        _any = isinstance(self._index_set, BASE.set._AnySet)
        if _any:
            validated_idx = _NotFound
        else:
            validated_idx = self._index_set.get(idx, _NotFound)
            if validated_idx is not _NotFound:
                return validated_idx
        if normalize_index.flatten:
            normalized_idx = normalize_index(idx)
            if normalized_idx is not idx and (not _any):
                if normalized_idx in self._data:
                    return normalized_idx
                if normalized_idx in self._index_set:
                    return normalized_idx
        else:
            normalized_idx = idx
        if normalized_idx.__class__ in slicer_types or (normalized_idx.__class__ is tuple and any((_.__class__ in slicer_types for _ in normalized_idx))):
            return self._processUnhashableIndex(normalized_idx)
        if _any:
            return idx
        if not self.is_indexed():
            raise KeyError("Cannot treat the scalar component '%s' as an indexed component" % (self.name,))
        raise KeyError("Index '%s' is not valid for indexed component '%s'" % (normalized_idx, self.name))

    def _processUnhashableIndex(self, idx):
        """Process a call to __getitem__ with unhashable elements

        There are three basic ways to get here:
          1) the index contains one or more slices or ellipsis
          2) the index contains an unhashable type (e.g., a Pyomo
             (Scalar)Component)
          3) the index contains an IndexTemplate
        """
        orig_idx = idx
        fixed = {}
        sliced = {}
        ellipsis = None
        if normalize_index.flatten:
            idx = normalize_index(idx)
        if idx.__class__ is not tuple:
            idx = (idx,)
        for i, val in enumerate(idx):
            if type(val) is slice:
                if val.start is not None or val.stop is not None or val.step is not None:
                    raise IndexError('Indexed components can only be indexed with simple slices: start and stop values are not allowed.')
                else:
                    if ellipsis is None:
                        sliced[i] = val
                    else:
                        sliced[i - len(idx)] = val
                    continue
            if val is Ellipsis:
                if ellipsis is not None:
                    raise IndexError("Indexed components can only be indexed with simple slices: the Pyomo wildcard slice (Ellipsis; e.g., '...') can only appear once")
                ellipsis = i
                continue
            if hasattr(val, 'is_expression_type'):
                _num_val = val
                try:
                    val = EXPR.evaluate_expression(val, constant=True)
                except TemplateExpressionError:
                    return EXPR.GetItemExpression((self,) + tuple(idx))
                except EXPR.NonConstantExpressionError:
                    raise RuntimeError('Error retrieving the value of an indexed item %s:\nindex %s is not a constant value.  This is likely not what you meant to\ndo, as if you later change the fixed value of the object this lookup\nwill not change.  If you understand the implications of using\nnon-constant values, you can get the current value of the object using\nthe value() function.' % (self.name, i))
                except EXPR.FixedExpressionError:
                    raise RuntimeError('Error retrieving the value of an indexed item %s:\nindex %s is a fixed but not constant value.  This is likely not what you\nmeant to do, as if you later change the fixed value of the object this\nlookup will not change.  If you understand the implications of using\nfixed but not constant values, you can get the current value using the\nvalue() function.' % (self.name, i))
            hash(val)
            if ellipsis is None:
                fixed[i] = val
            else:
                fixed[i - len(idx)] = val
        if sliced or ellipsis is not None:
            slice_dim = len(idx)
            if ellipsis is not None:
                slice_dim -= 1
            if normalize_index.flatten:
                set_dim = self.dim()
            elif not self.is_indexed():
                set_dim = 0
            else:
                set_dim = self.index_set().dimen
                if set_dim is None:
                    set_dim = 1
            structurally_valid = False
            if slice_dim == set_dim or set_dim is None:
                structurally_valid = True
            elif type(set_dim) is type:
                pass
            elif ellipsis is not None and slice_dim < set_dim:
                structurally_valid = True
            elif set_dim == 0 and idx == (slice(None),):
                structurally_valid = True
            if not structurally_valid:
                msg = "Index %s contains an invalid number of entries for component '%s'. Expected %s, got %s."
                if type(set_dim) is type:
                    set_dim = set_dim.__name__
                    msg += '\n    ' + '\n    '.join(textwrap.wrap(textwrap.dedent("\n                                Slicing components relies on knowing the\n                                underlying set dimensionality (even if the\n                                dimensionality is None).  The underlying\n                                component set ('%s') dimensionality has not been\n                                determined (likely because it is an empty Set).\n                                You can avoid this error by specifying the Set\n                                dimensionality (with the 'dimen=' keyword)." % (self.index_set(),)).strip()))
                raise IndexError(msg % (IndexedComponent_slice._getitem_args_to_str(list(idx)), self.name, set_dim, slice_dim))
            return IndexedComponent_slice(self, fixed, sliced, ellipsis)
        elif len(idx) == len(fixed):
            if len(idx) == 1:
                return fixed[0]
            else:
                return tuple((fixed[i] for i in range(len(idx))))
        else:
            raise DeveloperError(f"Unknown problem encountered when trying to retrieve index '{orig_idx}' for component '{self.name}'")

    def _getitem_when_not_present(self, index):
        """Returns/initializes a value when the index is not in the _data dict.

        Override this method if the component allows implicit member
        construction.  For classes that do not support a 'default' (at
        this point, everything except Param and Var), requesting
        _getitem_when_not_present will generate a KeyError (just like a
        normal dict).

        Implementations may assume that the index has already been validated
        and is a legitimate entry in the _data dict.

        """
        raise KeyError(index)

    def _setitem_impl(self, index, obj, value):
        """Perform the fundamental object value storage

        Components that want to implement a nonstandard storage mechanism
        should override this method.

        Implementations may assume that the index has already been
        validated and is a legitimate pre-existing entry in the _data
        dict.

        """
        if value is IndexedComponent.Skip:
            del self[index]
            return None
        else:
            obj.set_value(value)
        return obj

    def _setitem_when_not_present(self, index, value=_NotSpecified):
        """Perform the fundamental component item creation and storage.

        Components that want to implement a nonstandard storage mechanism
        should override this method.

        Implementations may assume that the index has already been
        validated and is a legitimate entry to add to the _data dict.
        """
        if value is IndexedComponent.Skip:
            return None
        if index is None and (not self.is_indexed()):
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)
        obj._index = index
        try:
            if value is not _NotSpecified:
                obj.set_value(value)
        except:
            self._data.pop(index, None)
            raise
        return obj

    def set_value(self, value):
        """Set the value of a scalar component."""
        if self.is_indexed():
            raise ValueError("Cannot set the value for the indexed component '%s' without specifying an index value.\n\tFor example, model.%s[i] = value" % (self.name, self.name))
        else:
            raise DeveloperError('Derived component %s failed to define set_value() for scalar instances.' % (self.__class__.__name__,))

    def _pprint(self):
        """Print component information."""
        return ([('Size', len(self)), ('Index', self._index_set if self.is_indexed() else None)], self._data.items(), ('Object',), lambda k, v: [type(v)])

    def id_index_map(self):
        """
        Return an dictionary id->index for
        all ComponentData instances.
        """
        result = {}
        for index, component_data in self.items():
            result[id(component_data)] = index
        return result