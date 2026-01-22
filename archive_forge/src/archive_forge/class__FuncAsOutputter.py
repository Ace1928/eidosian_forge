import copy
from typing import Any, Callable, Dict, List, Optional, no_type_check
from triad import ParamDict, to_uuid
from triad.utils.convert import get_caller_global_local_vars, to_function, to_instance
from fugue._utils.registry import fugue_plugin
from fugue.dataframe import DataFrames
from fugue.dataframe.function_wrapper import DataFrameFunctionWrapper
from fugue.exceptions import FugueInterfacelessError
from fugue.extensions._utils import (
from fugue.extensions.outputter.outputter import Outputter
class _FuncAsOutputter(Outputter):

    @property
    def validation_rules(self) -> Dict[str, Any]:
        return self._validation_rules

    @no_type_check
    def process(self, dfs: DataFrames) -> None:
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}
        if self._engine_param is not None:
            args.append(self._engine_param.to_input(self.execution_engine))
        if self._use_dfs:
            args.append(dfs)
        elif not dfs.has_key:
            args += dfs.values()
        else:
            kwargs.update(dfs)
        kwargs.update(self.params)
        return self._wrapper.run(args=args, kwargs=kwargs, ctx=self.execution_engine)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapper(*args, **kwargs)

    @no_type_check
    def __uuid__(self) -> str:
        return to_uuid(self._wrapper, self._engine_param, self._use_dfs)

    @no_type_check
    @staticmethod
    def from_func(func: Callable, validation_rules: Dict[str, Any]) -> '_FuncAsOutputter':
        validation_rules.update(parse_validation_rules_from_comment(func))
        tr = _FuncAsOutputter()
        tr._wrapper = DataFrameFunctionWrapper(func, '^e?(c|[dlspq]+)x*z?$', '^n$')
        tr._engine_param = tr._wrapper._params.get_value_by_index(0) if tr._wrapper.input_code.startswith('e') else None
        tr._use_dfs = 'c' in tr._wrapper.input_code
        tr._validation_rules = validation_rules
        return tr