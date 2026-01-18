from __future__ import annotations as _annotations
import sys
import warnings
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Set, Tuple, Type, Union
from typing_extensions import deprecated
def with_info_wrap_validator_function(function: WithInfoWrapValidatorFunction, schema: CoreSchema, *, field_name: str | None=None, ref: str | None=None, metadata: Any=None, serialization: SerSchema | None=None) -> WrapValidatorFunctionSchema:
    """
    Returns a schema which calls a function with a `validator` callable argument which can
    optionally be used to call inner validation with the function logic, this is much like the
    "onion" implementation of middleware in many popular web frameworks, an `info` argument is also passed, e.g.:

    ```py
    from pydantic_core import SchemaValidator, core_schema

    def fn(
        v: str,
        validator: core_schema.ValidatorFunctionWrapHandler,
        info: core_schema.ValidationInfo,
    ) -> str:
        return validator(input_value=v) + 'world'

    schema = core_schema.with_info_wrap_validator_function(
        function=fn, schema=core_schema.str_schema()
    )
    v = SchemaValidator(schema)
    assert v.validate_python('hello ') == 'hello world'
    ```

    Args:
        function: The validator function to call
        schema: The schema to validate the output of the validator function
        field_name: The name of the field this validators is applied to, if any
        ref: optional unique identifier of the schema, used to reference the schema in other places
        metadata: Any other information you want to include with the schema, not used by pydantic-core
        serialization: Custom serialization schema
    """
    return _dict_not_none(type='function-wrap', function=_dict_not_none(type='with-info', function=function, field_name=field_name), schema=schema, ref=ref, metadata=metadata, serialization=serialization)