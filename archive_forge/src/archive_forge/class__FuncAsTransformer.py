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
class _FuncAsTransformer(Transformer):

    def validate_on_compile(self) -> None:
        super().validate_on_compile()
        _validate_callback(self)

    def get_output_schema(self, df: DataFrame) -> Any:
        return self._parse_schema(self._output_schema_arg, df)

    def get_format_hint(self) -> Optional[str]:
        return self._format_hint

    @property
    def validation_rules(self) -> Dict[str, Any]:
        return self._validation_rules

    @no_type_check
    def transform(self, df: LocalDataFrame) -> LocalDataFrame:
        args = [df] + _get_callback(self)
        return self._wrapper.run(args, self.params, ignore_unknown=False, output_schema=self.output_schema)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapper(*args, **kwargs)

    @no_type_check
    def __uuid__(self) -> str:
        return to_uuid(self._wrapper.__uuid__(), self._output_schema_arg)

    def _parse_schema(self, obj: Any, df: DataFrame) -> Schema:
        if callable(obj):
            return obj(df, **self.params)
        if isinstance(obj, str):
            return df.schema.transform(obj)
        if isinstance(obj, List):
            return df.schema.transform(*obj)
        raise NotImplementedError

    @staticmethod
    def from_func(func: Callable, schema: Any, validation_rules: Dict[str, Any]) -> '_FuncAsTransformer':
        if schema is None:
            schema = parse_output_schema_from_comment(func)
        if isinstance(schema, Schema):
            schema = str(schema)
        validation_rules.update(parse_validation_rules_from_comment(func))
        assert_arg_not_none(schema, 'schema')
        tr = _FuncAsTransformer()
        tr._wrapper = DataFrameFunctionWrapper(func, '^[lspq][fF]?x*z?$', '^[lspq]$')
        tr._output_schema_arg = schema
        tr._validation_rules = validation_rules
        tr._uses_callback = 'f' in tr._wrapper.input_code.lower()
        tr._requires_callback = 'F' in tr._wrapper.input_code
        tr._format_hint = tr._wrapper.get_format_hint()
        return tr