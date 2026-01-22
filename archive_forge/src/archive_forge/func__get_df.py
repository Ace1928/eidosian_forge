import re
from typing import Any, Dict, Optional
from triad import assert_or_throw
from ..collections.yielded import Yielded
from ..exceptions import FugueSQLError
from ..workflow.workflow import FugueWorkflow, WorkflowDataFrame
def _get_df(self) -> WorkflowDataFrame:
    if isinstance(self._df, Yielded):
        return self._workflow.df(self._df)
    if isinstance(self._df, WorkflowDataFrame):
        assert_or_throw(self._df.workflow is self._workflow, lambda: FugueSQLError(f'{self._key}, {self._df} is from another workflow'))
        return self._df
    return self._workflow.df(self._df)