from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from .bases import Property
class PandasDataFrame(Property['DataFrame']):
    """ Accept Pandas DataFrame values.

    This property only exists to support type validation, e.g. for "accepts"
    clauses. It is not serializable itself, and is not useful to add to
    Bokeh models directly.

    """

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, detail)
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            return
        msg = '' if not detail else f'expected Pandas DataFrame, got {value!r}'
        raise ValueError(msg)