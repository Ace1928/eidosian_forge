import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
def _to_dask(cls, modin_obj):
    """
        Write query compiler content to a Dask DataFrame/Series.

        Parameters
        ----------
        modin_obj : modin.pandas.DataFrame, modin.pandas.Series
            The Modin DataFrame/Series to write.

        Returns
        -------
        dask.dataframe.DataFrame or dask.dataframe.Series
            A Dask DataFrame/Series object.

        Notes
        -----
        Modin DataFrame/Series can only be converted to a Dask DataFrame/Series if Modin uses a Dask engine.
        """
    return cls.io_cls.to_dask(modin_obj)