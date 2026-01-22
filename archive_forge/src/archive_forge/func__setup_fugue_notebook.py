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
def _setup_fugue_notebook(ipython: Any, setup_obj: Any, fsql_ignore_case: bool=False) -> None:
    s = NotebookSetup() if setup_obj is None else to_instance(setup_obj, NotebookSetup)
    magics = _FugueSQLMagics(ipython, dict(s.get_pre_conf()), dict(s.get_post_conf()), fsql_ignore_case=fsql_ignore_case)
    ipython.register_magics(magics)