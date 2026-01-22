import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
def _to_ray(cls, modin_obj):
    """
        Write query compiler content to a Ray Dataset.

        Parameters
        ----------
        modin_obj : modin.pandas.DataFrame, modin.pandas.Series
            The Modin DataFrame/Series to write.

        Returns
        -------
        ray.data.Dataset
            A Ray Dataset object.

        Notes
        -----
        Modin DataFrame/Series can only be converted to a Ray Dataset if Modin uses a Ray engine.
        """
    return cls.io_cls.to_ray(modin_obj)