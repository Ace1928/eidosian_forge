import copy
from typing import Any, Callable, Dict, List, Optional, no_type_check
from triad import ParamDict
from triad.collections import Schema
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from triad.utils.hash import to_uuid
from fugue._utils.interfaceless import parse_output_schema_from_comment
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import DataFrame
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions.creator.creator import Creator
from .._utils import load_namespace_extensions
class _FuncAsCreator(Creator):

    @no_type_check
    def create(self) -> DataFrame:
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if self._engine_param is not None:
            args.append(self._engine_param.to_input(self.execution_engine))
        kwargs.update(self.params)
        return self._wrapper.run(args=args, kwargs=kwargs, output_schema=self.output_schema if self._need_output_schema else None, ctx=self.execution_engine)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapper(*args, **kwargs)

    @no_type_check
    def __uuid__(self) -> str:
        return to_uuid(self._wrapper, self._engine_param, self._need_output_schema, str(self._output_schema))

    @no_type_check
    @staticmethod
    def from_func(func: Callable, schema: Any) -> '_FuncAsCreator':
        if schema is None:
            schema = parse_output_schema_from_comment(func)
        tr = _FuncAsCreator()
        tr._wrapper = DataFrameFunctionWrapper(func, '^e?x*z?$', '^[dlspq]$')
        tr._engine_param = tr._wrapper._params.get_value_by_index(0) if tr._wrapper.input_code.startswith('e') else None
        tr._need_output_schema = tr._wrapper.need_output_schema
        tr._output_schema = Schema(schema)
        if len(tr._output_schema) == 0:
            assert_or_throw(tr._need_output_schema is None or not tr._need_output_schema, FugueInterfacelessError(f'schema must be provided for return type {tr._wrapper._rt}'))
        else:
            assert_or_throw(tr._need_output_schema is None or tr._need_output_schema, FugueInterfacelessError(f'schema must not be provided for return type {tr._wrapper._rt}'))
        return tr