from __future__ import annotations
import math
from datetime import date, timedelta
from typing import Literal, overload
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException
def time_to_seconds(t: float | timedelta | str | None, *, coerce_none_to_inf: bool=True) -> float | None:
    """
    Convert a time string value to a float representing "number of seconds".
    """
    if coerce_none_to_inf and t is None:
        return math.inf
    if isinstance(t, timedelta):
        return t.total_seconds()
    if isinstance(t, str):
        import numpy as np
        import pandas as pd
        try:
            seconds: float = pd.Timedelta(t).total_seconds()
            if np.isnan(seconds):
                raise BadTimeStringError(t)
            return seconds
        except ValueError as ex:
            raise BadTimeStringError(t) from ex
    return t