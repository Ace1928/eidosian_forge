import datetime
import pandas
import pytest
from pandas.core.dtypes.common import is_datetime64_any_dtype, is_object_dtype
import modin.pandas as pd
from modin.tests.pandas.utils import df_equals
from modin.tests.pandas.utils import eval_io as general_eval_io
from modin.utils import try_cast_to_pandas
class ForceHdkImport:
    """
    Trigger import execution for Modin DataFrames obtained by HDK engine if already not.

    When using as a context class also cleans up imported tables at the end of the context.

    Parameters
    ----------
    *dfs : iterable
        DataFrames to trigger import.
    """

    def __init__(self, *dfs):
        self._imported_frames = []
        for df in dfs:
            if not isinstance(df, (pd.DataFrame, pd.Series)):
                continue
            if df.empty:
                continue
            try:
                modin_frame = df._query_compiler._modin_frame
                modin_frame.force_import()
                self._imported_frames.append(df)
            except NotImplementedError:
                ...

    def __enter__(self):
        return self

    def export_frames(self):
        """
        Export tables from HDK that was imported by this instance.

        Returns
        -------
        list
            A list of Modin DataFrames whose payload is ``pyarrow.Table``
            that was just exported from HDK.
        """
        result = []
        for df in self._imported_frames:
            df = df[df.columns.tolist()]
            modin_frame = df._query_compiler._modin_frame
            mode = modin_frame._force_execution_mode
            modin_frame._force_execution_mode = 'hdk'
            modin_frame._execute()
            modin_frame._force_execution_mode = mode
            result.append(df)
        return result

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._imported_frames.clear()