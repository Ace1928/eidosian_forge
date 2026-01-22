import parso
import os
from inspect import Parameter
from jedi import debug
from jedi.inference.utils import safe_property
from jedi.inference.helpers import get_str_or_none
from jedi.inference.arguments import iterate_argument_clinic, ParamIssue, \
from jedi.inference import analysis
from jedi.inference import compiled
from jedi.inference.value.instance import \
from jedi.inference.base_value import ContextualizedNode, \
from jedi.inference.value import ClassValue, ModuleValue
from jedi.inference.value.klass import ClassMixin
from jedi.inference.value.function import FunctionMixin
from jedi.inference.value import iterable
from jedi.inference.lazy_value import LazyTreeValue, LazyKnownValue, \
from jedi.inference.names import ValueName, BaseTreeParamName
from jedi.inference.filters import AttributeOverwrite, publish_method, \
from jedi.inference.signature import AbstractSignature, SignatureWrapper
from operator import itemgetter as _itemgetter
from collections import OrderedDict
def _create_string_input_function(func):

    @argument_clinic('string, /', want_value=True, want_arguments=True)
    def wrapper(strings, value, arguments):

        def iterate():
            for value in strings:
                s = get_str_or_none(value)
                if s is not None:
                    s = func(s)
                    yield compiled.create_simple_object(value.inference_state, s)
        values = ValueSet(iterate())
        if values:
            return values
        return value.py__call__(arguments)
    return wrapper