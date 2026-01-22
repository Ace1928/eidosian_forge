import json
from typing import Any, Dict, List, Optional, Type, TypeVar
from adagio.exceptions import DependencyDefinitionError, DependencyNotDefinedError
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot, assert_arg_not_none
from triad.utils.convert import (
from triad.utils.hash import to_uuid
from triad.utils.string import assert_triad_var_name
class ConfigSpec(OutputSpec):

    def __init__(self, name: str, data_type: Any, nullable: bool, required: bool=True, default_value: Any=None, metadata: Any=None):
        super().__init__(name, data_type, nullable, metadata)
        self.required = required
        self.default_value = default_value
        if required:
            aot(default_value is None, "required var can't have default_value")
        elif default_value is None:
            aot(nullable, "default_value can't be None because it's not nullable")
        else:
            self.default_value = as_type(self.default_value, self.data_type)

    def validate_value(self, obj: Any) -> Any:
        if obj is not None:
            return super().validate_value(obj)
        aot(self.nullable, lambda: f"Can't set None to {self.paramdict}")
        return obj

    @property
    def attributes(self) -> List[str]:
        return ['name', 'data_type', 'nullable', 'required', 'default_value', 'metadata']