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
@argument_clinic('*args, /', want_callback=True)
def _os_path_join(args_set, callback):
    if len(args_set) == 1:
        string = ''
        sequence, = args_set
        is_first = True
        for lazy_value in sequence.py__iter__():
            string_values = lazy_value.infer()
            if len(string_values) != 1:
                break
            s = get_str_or_none(next(iter(string_values)))
            if s is None:
                break
            if not is_first:
                string += os.path.sep
            string += s
            is_first = False
        else:
            return ValueSet([compiled.create_simple_object(sequence.inference_state, string)])
    return callback()