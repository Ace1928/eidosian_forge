import html
import json
from typing import Any, Dict, List, Optional
from IPython import get_ipython
from IPython.core.magic import Magics, cell_magic, magics_class, needs_local_scope
from IPython.display import HTML, display
from triad import ParamDict
from triad.utils.convert import to_instance
from triad.utils.pyarrow import _field_to_expression
from fugue import DataFrame, DataFrameDisplay, ExecutionEngine
from fugue import fsql as fugue_sql
from fugue import get_dataset_display, make_execution_engine
from fugue.dataframe import YieldedDataFrame
from fugue.exceptions import FugueSQLSyntaxError
@magics_class
class _FugueSQLMagics(Magics):
    """Fugue SQL Magics"""

    def __init__(self, shell: Any, pre_conf: Dict[str, Any], post_conf: Dict[str, Any], fsql_ignore_case: bool=False):
        super().__init__(shell)
        self._pre_conf = pre_conf
        self._post_conf = post_conf
        self._fsql_ignore_case = fsql_ignore_case

    @needs_local_scope
    @cell_magic('fsql')
    def fsql(self, line: str, cell: str, local_ns: Any=None) -> None:
        try:
            dag = fugue_sql('\n' + cell, local_ns, fsql_ignore_case=self._fsql_ignore_case)
        except FugueSQLSyntaxError as ex:
            raise FugueSQLSyntaxError(str(ex)).with_traceback(None) from None
        dag.run(self.get_engine(line, {} if local_ns is None else local_ns))
        for k, v in dag.yields.items():
            if isinstance(v, YieldedDataFrame):
                local_ns[k] = v.result
            else:
                local_ns[k] = v

    def get_engine(self, line: str, lc: Dict[str, Any]) -> ExecutionEngine:
        line = line.strip()
        p = line.find('{')
        if p >= 0:
            engine = line[:p].strip()
            conf = json.loads(line[p:])
        else:
            parts = line.split(' ', 1)
            engine = parts[0]
            conf = ParamDict(None if len(parts) == 1 else lc[parts[1]])
        cf = dict(self._pre_conf)
        cf.update(conf)
        for k, v in self._post_conf.items():
            if k in cf and cf[k] != v:
                raise ValueError(f'{k} must be {v}, but you set to {cf[k]}, you may unset it')
            cf[k] = v
        if '+' in engine:
            return make_execution_engine(tuple(engine.split('+', 1)), cf)
        return make_execution_engine(engine, cf)