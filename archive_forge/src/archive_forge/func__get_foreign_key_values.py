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
@iterator_to_value_set
def _get_foreign_key_values(cls, field_tree_instance):
    if isinstance(field_tree_instance, TreeInstance):
        argument_iterator = field_tree_instance._arguments.unpack()
        key, lazy_values = next(argument_iterator, (None, None))
        if key is None and lazy_values is not None:
            for value in lazy_values.infer():
                if value.py__name__() == 'str':
                    foreign_key_class_name = value.get_safe_value()
                    module = cls.get_root_context()
                    for v in module.py__getattribute__(foreign_key_class_name):
                        if v.is_class():
                            yield v
                elif value.is_class():
                    yield value