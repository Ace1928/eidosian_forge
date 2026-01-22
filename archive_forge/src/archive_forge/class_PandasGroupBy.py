from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from .bases import Property
class PandasGroupBy(Property['GroupBy[Any]']):
    """ Accept Pandas DataFrame values.

    This property only exists to support type validation, e.g. for "accepts"
    clauses. It is not serializable itself, and is not useful to add to
    Bokeh models directly.

    """

    def validate(self, value: Any, detail: bool=True) -> None:
        super().validate(value, detail)
        from pandas.core.groupby import GroupBy
        if isinstance(value, GroupBy):
            return
        msg = '' if not detail else f'expected Pandas GroupBy, got {value!r}'
        raise ValueError(msg)