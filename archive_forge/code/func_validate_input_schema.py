from typing import Any, Callable, Dict
from triad import Schema
from triad.utils.assertion import assert_or_throw
from fugue._utils.interfaceless import parse_comment_annotation
from fugue.collections.partition import PartitionSpec, parse_presort_exp
from fugue.exceptions import (
def validate_input_schema(schema: Schema, rules: Dict[str, Any]) -> None:
    for k, v in rules.items():
        if k == 'input_has':
            for x in v:
                assert_or_throw(x in schema, lambda: FugueWorkflowRuntimeValidationError(f'required column {x} is not in {schema}'))
        if k == 'input_is':
            assert_or_throw(schema == v, lambda: FugueWorkflowRuntimeValidationError(f'{v} does not match {schema}'))