from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineError
from ..mapping.evaluation import after_stat
from ..scales.scale_continuous import scale_continuous
from .stat import stat

    Superimpose a function onto a plot

    {usage}

    Parameters
    ----------
    {common_parameters}
    fun : callable
        Function to evaluate.
    n : int, default=101
        Number of points at which to evaluate the function.
    xlim : tuple, default=None
        `x` limits for the range. The default depends on
        the `x` aesthetic. There is not an `x` aesthetic
        then the `xlim` must be provided.
    args : Optional[tuple[Any] | dict[str, Any]], default=None
        Arguments to pass to `fun`.
    