import logging
from weakref import ref as weakref_ref
from pyomo.common.log import is_debug_set
from pyomo.core.base.set_types import Any
from pyomo.core.base.var import IndexedVar, _VarData
from pyomo.core.base.constraint import IndexedConstraint, _ConstraintData
from pyomo.core.base.objective import IndexedObjective, _ObjectiveData
from pyomo.core.base.expression import IndexedExpression, _ExpressionData
from collections.abc import MutableMapping
from collections.abc import Mapping
class ComponentDict(MutableMapping):

    def __init__(self, interface_datatype, *args):
        self._interface_datatype = interface_datatype
        self._data = {}
        if len(args) > 0:
            if len(args) > 1:
                raise TypeError('ComponentDict expected at most 1 arguments, got %s' % len(args))
            self.update(args[0])

    def construct(self, data=None):
        if is_debug_set(logger):
            logger.debug('Constructing ComponentDict object, name=%s, from data=%s' % (self.name, str(data)))
        if self._constructed:
            return
        self._constructed = True

    def __setitem__(self, key, val):
        if isinstance(val, self._interface_datatype):
            if val._component is None:
                val._component = weakref_ref(self)
                if hasattr(self, '_active'):
                    self._active |= getattr(val, '_active', True)
                if key in self._data:
                    self._data[key]._component = None
                self._data[key] = val
                self._data[key]._index = key
                return
            elif key in self._data and self._data[key] is val:
                return
            raise ValueError('Invalid component object assignment to ComponentDict %s at key %s. A parent component has already been assigned the object: %s' % (self.name, key, val.parent_component().name))
        raise TypeError('ComponentDict must be assigned objects of type %s. Invalid type for key %s: %s' % (self._interface_datatype.__name__, key, type(val)))

    def __delitem__(self, key):
        obj = self._data[key]
        obj._component = None
        del self._data[key]

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return self._data.__len__()

    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return False
        return dict(((key, (type(val), id(val))) for key, val in self.items())) == dict(((key, (type(val), id(val))) for key, val in other.items()))

    def __ne__(self, other):
        return not self == other