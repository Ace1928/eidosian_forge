from __future__ import annotations
from collections.abc import (
from typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.common import (
def validate_freeze_panes(freeze_panes: tuple[int, int] | None) -> bool:
    if freeze_panes is not None:
        if len(freeze_panes) == 2 and all((isinstance(item, int) for item in freeze_panes)):
            return True
        raise ValueError('freeze_panes must be of form (row, column) where row and column are integers')
    return False