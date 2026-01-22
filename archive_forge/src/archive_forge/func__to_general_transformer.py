import copy
from typing import Any, Callable, Dict, List, Optional, Type, Union, no_type_check
from triad import ParamDict, Schema
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from triad.utils.hash import to_uuid
from fugue._utils.interfaceless import is_class_method, parse_output_schema_from_comment
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import ArrayDataFrame, DataFrame, DataFrames, LocalDataFrame
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions.transformer.constants import OUTPUT_TRANSFORMER_DUMMY_SCHEMA
from fugue.extensions.transformer.transformer import CoTransformer, Transformer
from .._utils import (
def _to_general_transformer(obj: Any, schema: Any, global_vars: Optional[Dict[str, Any]], local_vars: Optional[Dict[str, Any]], validation_rules: Optional[Dict[str, Any]], func_transformer_type: Type, func_cotransformer_type: Type) -> Union[Transformer, CoTransformer]:
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
    exp: Optional[Exception] = None
    if validation_rules is None:
        validation_rules = {}
    try:
        return copy.copy(to_instance(obj, Transformer, global_vars=global_vars, local_vars=local_vars))
    except Exception as e:
        exp = e
    try:
        return copy.copy(to_instance(obj, CoTransformer, global_vars=global_vars, local_vars=local_vars))
    except Exception as e:
        exp = e
    try:
        f = to_function(obj, global_vars=global_vars, local_vars=local_vars)
        if isinstance(f, Transformer):
            return copy.copy(f)
        return func_transformer_type.from_func(f, schema, validation_rules=validation_rules)
    except Exception as e:
        exp = e
    try:
        f = to_function(obj, global_vars=global_vars, local_vars=local_vars)
        if isinstance(f, CoTransformer):
            return copy.copy(f)
        return func_cotransformer_type.from_func(f, schema, validation_rules=validation_rules)
    except Exception as e:
        exp = e
    raise FugueInterfacelessError(f'{obj} is not a valid transformer', exp)