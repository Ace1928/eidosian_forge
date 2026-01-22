from __future__ import annotations as _annotations
from inspect import Parameter, signature
from typing import Any, Dict, Tuple, Union, cast
from pydantic_core import core_schema
from typing_extensions import Protocol
from ..errors import PydanticUserError
from ._decorators import can_be_positional
def _wrapper2(fields_tuple: RootValidatorFieldsTuple, _: core_schema.ValidationInfo) -> RootValidatorFieldsTuple:
    if len(fields_tuple) == 2:
        values, init_vars = fields_tuple
        values = validator(values)
        return (values, init_vars)
    else:
        model_dict, model_extra, fields_set = fields_tuple
        if model_extra:
            fields = set(model_dict.keys())
            model_dict.update(model_extra)
            model_dict_new = validator(model_dict)
            for k in list(model_dict_new.keys()):
                if k not in fields:
                    model_extra[k] = model_dict_new.pop(k)
        else:
            model_dict_new = validator(model_dict)
        return (model_dict_new, model_extra, fields_set)