import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
def _to_pickle(cls, *args, **kwargs):
    """
        Pickle query compiler object.

        Parameters
        ----------
        *args : args
            Arguments to the writer method.
        **kwargs : kwargs
            Arguments to the writer method.
        """
    return cls.io_cls.to_pickle(*args, **kwargs)