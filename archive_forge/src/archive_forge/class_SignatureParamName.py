import re
from functools import partial
from inspect import Parameter
from pathlib import Path
from typing import Optional
from jedi import debug
from jedi.inference.utils import to_list
from jedi.cache import memoize_method
from jedi.inference.filters import AbstractFilter
from jedi.inference.names import AbstractNameDefinition, ValueNameMixin, \
from jedi.inference.base_value import Value, ValueSet, NO_VALUES
from jedi.inference.lazy_value import LazyKnownValue
from jedi.inference.compiled.access import _sentinel
from jedi.inference.cache import inference_state_function_cache
from jedi.inference.helpers import reraise_getitem_errors
from jedi.inference.signature import BuiltinSignature
from jedi.inference.context import CompiledContext, CompiledModuleContext
class SignatureParamName(ParamNameInterface, AbstractNameDefinition):

    def __init__(self, compiled_value, signature_param):
        self.parent_context = compiled_value.parent_context
        self._signature_param = signature_param

    @property
    def string_name(self):
        return self._signature_param.name

    def to_string(self):
        s = self._kind_string() + self.string_name
        if self._signature_param.has_annotation:
            s += ': ' + self._signature_param.annotation_string
        if self._signature_param.has_default:
            s += '=' + self._signature_param.default_string
        return s

    def get_kind(self):
        return getattr(Parameter, self._signature_param.kind_name)

    def infer(self):
        p = self._signature_param
        inference_state = self.parent_context.inference_state
        values = NO_VALUES
        if p.has_default:
            values = ValueSet([create_from_access_path(inference_state, p.default)])
        if p.has_annotation:
            annotation = create_from_access_path(inference_state, p.annotation)
            values |= annotation.execute_with_values()
        return values