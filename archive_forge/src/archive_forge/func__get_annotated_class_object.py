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
@inference_state_method_cache(default=None)
def _get_annotated_class_object(self):
    from jedi.inference.gradual.annotation import py__annotations__, infer_type_vars_for_execution
    args = InstanceArguments(self, self._arguments)
    for signature in self.class_value.py__getattribute__('__init__').get_signatures():
        funcdef = signature.value.tree_node
        if funcdef is None or funcdef.type != 'funcdef' or (not signature.matches_signature(args)):
            continue
        bound_method = BoundMethod(self, self.class_value.as_context(), signature.value)
        all_annotations = py__annotations__(funcdef)
        type_var_dict = infer_type_vars_for_execution(bound_method, args, all_annotations)
        if type_var_dict:
            defined, = self.class_value.define_generics(infer_type_vars_for_execution(signature.value, args, all_annotations))
            debug.dbg('Inferred instance value as %s', defined, color='BLUE')
            return defined
    return None