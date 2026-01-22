from abc import abstractproperty
from parso.tree import search_ancestor
from jedi import debug
from jedi import settings
from jedi.inference import compiled
from jedi.inference.compiled.value import CompiledValueFilter
from jedi.inference.helpers import values_from_qualified_names, is_big_annoying_library
from jedi.inference.filters import AbstractFilter, AnonymousFunctionExecutionFilter
from jedi.inference.names import ValueName, TreeNameDefinition, ParamName, \
from jedi.inference.base_value import Value, NO_VALUES, ValueSet, \
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.arguments import ValuesArguments, TreeArgumentsWrapper
from jedi.inference.value.function import \
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.dynamic_arrays import get_dynamic_array_instance
from jedi.parser_utils import function_is_staticmethod, function_is_classmethod
class AbstractInstanceValue(Value):
    api_type = 'instance'

    def __init__(self, inference_state, parent_context, class_value):
        super().__init__(inference_state, parent_context)
        self.class_value = class_value

    def is_instance(self):
        return True

    def get_qualified_names(self):
        return self.class_value.get_qualified_names()

    def get_annotated_class_object(self):
        return self.class_value

    def py__class__(self):
        return self.class_value

    def py__bool__(self):
        return None

    @abstractproperty
    def name(self):
        raise NotImplementedError

    def get_signatures(self):
        call_funcs = self.py__getattribute__('__call__').py__get__(self, self.class_value)
        return [s.bind(self) for s in call_funcs.get_signatures()]

    def get_function_slot_names(self, name):
        for filter in self.get_filters(include_self_names=False):
            names = filter.get(name)
            if names:
                return names
        return []

    def execute_function_slots(self, names, *inferred_args):
        return ValueSet.from_sets((name.infer().execute_with_values(*inferred_args) for name in names))

    def get_type_hint(self, add_class_info=True):
        return self.py__name__()

    def py__getitem__(self, index_value_set, contextualized_node):
        names = self.get_function_slot_names('__getitem__')
        if not names:
            return super().py__getitem__(index_value_set, contextualized_node)
        args = ValuesArguments([index_value_set])
        return ValueSet.from_sets((name.infer().execute(args) for name in names))

    def py__iter__(self, contextualized_node=None):
        iter_slot_names = self.get_function_slot_names('__iter__')
        if not iter_slot_names:
            return super().py__iter__(contextualized_node)

        def iterate():
            for generator in self.execute_function_slots(iter_slot_names):
                yield from generator.py__next__(contextualized_node)
        return iterate()

    def __repr__(self):
        return '<%s of %s>' % (self.__class__.__name__, self.class_value)