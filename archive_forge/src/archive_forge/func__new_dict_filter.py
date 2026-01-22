from inspect import Parameter
from jedi import debug
from jedi.inference.cache import inference_state_function_cache
from jedi.inference.base_value import ValueSet, iterator_to_value_set, ValueWrapper
from jedi.inference.filters import DictFilter, AttributeOverwrite
from jedi.inference.names import NameWrapper, BaseTreeParamName
from jedi.inference.compiled.value import EmptyCompiledName
from jedi.inference.value.instance import TreeInstance
from jedi.inference.value.klass import ClassMixin
from jedi.inference.gradual.base import GenericClass
from jedi.inference.gradual.generics import TupleGenericManager
from jedi.inference.signature import AbstractSignature
def _new_dict_filter(cls, is_instance):
    filters = list(cls.get_filters(is_instance=is_instance, include_metaclasses=False, include_type_when_class=False))
    dct = {name.string_name: DjangoModelName(cls, name, is_instance) for filter_ in reversed(filters) for name in filter_.values()}
    if is_instance:
        dct['objects'] = EmptyCompiledName(cls.inference_state, 'objects')
    return DictFilter(dct)