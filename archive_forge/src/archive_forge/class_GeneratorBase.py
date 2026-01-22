from jedi.inference import compiled
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues, \
from jedi.inference.helpers import get_int_or_none, is_string, \
from jedi.inference.utils import safe_property, to_list
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.filters import LazyAttributeOverwrite, publish_method
from jedi.inference.base_value import ValueSet, Value, NO_VALUES, \
from jedi.parser_utils import get_sync_comp_fors
from jedi.inference.context import CompForContext
from jedi.inference.value.dynamic_arrays import check_array_additions
class GeneratorBase(LazyAttributeOverwrite, IterableMixin):
    array_type = None

    def _get_wrapped_value(self):
        instance, = self._get_cls().execute_annotation()
        return instance

    def _get_cls(self):
        generator, = self.inference_state.typing_module.py__getattribute__('Generator')
        return generator

    def py__bool__(self):
        return True

    @publish_method('__iter__')
    def _iter(self, arguments):
        return ValueSet([self])

    @publish_method('send')
    @publish_method('__next__')
    def _next(self, arguments):
        return ValueSet.from_sets((lazy_value.infer() for lazy_value in self.py__iter__()))

    def py__stop_iteration_returns(self):
        return ValueSet([compiled.builtin_from_name(self.inference_state, 'None')])

    @property
    def name(self):
        return compiled.CompiledValueName(self, 'Generator')

    def get_annotated_class_object(self):
        from jedi.inference.gradual.generics import TupleGenericManager
        gen_values = self.merge_types_of_iterate().py__class__()
        gm = TupleGenericManager((gen_values, NO_VALUES, NO_VALUES))
        return self._get_cls().with_generics(gm)