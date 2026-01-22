import re
from itertools import zip_longest
from parso.python import tree
from jedi import debug
from jedi.inference.utils import PushBackIterator
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues, \
from jedi.inference.names import ParamName, TreeNameDefinition, AnonymousParamName
from jedi.inference.base_value import NO_VALUES, ValueSet, ContextualizedNode
from jedi.inference.value import iterable
from jedi.inference.cache import inference_state_as_method_param_cache
class ValuesArguments(AbstractArguments):

    def __init__(self, values_list):
        self._values_list = values_list

    def unpack(self, funcdef=None):
        for values in self._values_list:
            yield (None, LazyKnownValues(values))

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self._values_list)