from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ValueSet, NO_VALUES, Value, \
from jedi.inference.compiled import builtin_from_name
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.klass import ClassMixin
from jedi.inference.utils import to_list
from jedi.inference.names import AbstractNameDefinition, ValueName
from jedi.inference.context import ClassContext
from jedi.inference.gradual.generics import TupleGenericManager
class _LazyGenericBaseClass:

    def __init__(self, class_value, lazy_base_class, generics_manager):
        self._class_value = class_value
        self._lazy_base_class = lazy_base_class
        self._generics_manager = generics_manager

    @iterator_to_value_set
    def infer(self):
        for base in self._lazy_base_class.infer():
            if isinstance(base, GenericClass):
                yield GenericClass.create_cached(base.inference_state, base._wrapped_value, TupleGenericManager(tuple(self._remap_type_vars(base))))
            elif base.is_class_mixin():
                yield GenericClass.create_cached(base.inference_state, base, self._generics_manager)
            else:
                yield base

    def _remap_type_vars(self, base):
        from jedi.inference.gradual.type_var import TypeVar
        filter = self._class_value.get_type_var_filter()
        for type_var_set in base.get_generics():
            new = NO_VALUES
            for type_var in type_var_set:
                if isinstance(type_var, TypeVar):
                    names = filter.get(type_var.py__name__())
                    new |= ValueSet.from_sets((name.infer() for name in names))
                else:
                    new |= ValueSet([type_var])
            yield new

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self._lazy_base_class)