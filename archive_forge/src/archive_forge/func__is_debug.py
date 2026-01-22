import logging
import uuid
from abc import ABC
from copy import copy
import pandas
from pandas.api.types import is_scalar
from pandas.util import cache_readonly
from modin.core.storage_formats.pandas.utils import length_fn_pandas, width_fn_pandas
from modin.logging import ClassLogger, get_logger
from modin.pandas.indexing import compute_sliced_len
def _is_debug(self, logger=None):
    """
        Check that the logger is set to debug mode.

        Parameters
        ----------
        logger : logging.logger, optional
            Logger obtained from Modin's `get_logger` utility.
            Explicit transmission of this parameter can be used in the case
            when within the context of `_is_debug` call there was already
            `get_logger` call. This is an optimization.

        Returns
        -------
        bool
        """
    if logger is None:
        logger = get_logger()
    return logger.isEnabledFor(logging.DEBUG)