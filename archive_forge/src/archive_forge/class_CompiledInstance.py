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
class CompiledInstance(AbstractInstanceValue):

    def __init__(self, inference_state, parent_context, class_value, arguments):
        super().__init__(inference_state, parent_context, class_value)
        self._arguments = arguments

    def get_filters(self, origin_scope=None, include_self_names=True):
        class_value = self.get_annotated_class_object()
        class_filters = class_value.get_filters(origin_scope=origin_scope, is_instance=True)
        for f in class_filters:
            yield CompiledInstanceClassFilter(self, f)

    @property
    def name(self):
        return compiled.CompiledValueName(self, self.class_value.name.string_name)

    def is_stub(self):
        return False