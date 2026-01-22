import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
def _to_sql(cls, *args, **kwargs):
    """
        Write query compiler content to a SQL database.

        Parameters
        ----------
        *args : args
            Arguments to the writer method.
        **kwargs : kwargs
            Arguments to the writer method.
        """
    return cls.io_cls.to_sql(*args, **kwargs)