from parso.python import tree
from jedi import debug
from jedi.inference.cache import inference_state_method_cache, CachedMetaClass
from jedi.inference import compiled
from jedi.inference import recursion
from jedi.inference import docstrings
from jedi.inference import flow_analysis
from jedi.inference.signature import TreeSignature
from jedi.inference.filters import ParserTreeFilter, FunctionExecutionFilter, \
from jedi.inference.names import ValueName, AbstractNameDefinition, \
from jedi.inference.base_value import ContextualizedNode, NO_VALUES, \
from jedi.inference.lazy_value import LazyKnownValues, LazyKnownValue, \
from jedi.inference.context import ValueContext, TreeContextMixin
from jedi.inference.value import iterable
from jedi import parser_utils
from jedi.inference.parser_cache import get_yield_exprs
from jedi.inference.helpers import values_from_qualified_names
from jedi.inference.gradual.generics import TupleGenericManager
class MethodValue(FunctionValue):

    def __init__(self, inference_state, class_context, *args, **kwargs):
        super().__init__(inference_state, *args, **kwargs)
        self.class_context = class_context

    def get_default_param_context(self):
        return self.class_context

    def get_qualified_names(self):
        names = self.class_context.get_qualified_names()
        if names is None:
            return None
        return names + (self.py__name__(),)

    @property
    def name(self):
        return FunctionNameInClass(self.class_context, super().name)