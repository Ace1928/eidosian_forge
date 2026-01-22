from typing import Any, Callable, Optional
from triad import Schema, assert_or_throw, to_uuid
from fugue.collections.yielded import Yielded
from fugue.dataframe import DataFrame
from fugue.exceptions import FugueWorkflowCompileError
from fugue.execution.api import as_fugue_engine_df
from fugue.extensions.creator import Creator
def _df_uid(self):
    if self._data_determiner is not None:
        return self._data_determiner(self._df)
    if isinstance(self._df, Yielded):
        return self._df
    return 1