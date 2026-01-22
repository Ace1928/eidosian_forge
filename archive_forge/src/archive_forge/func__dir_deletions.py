from __future__ import annotations
from typing import (
import warnings
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
@final
def _dir_deletions(self) -> set[str]:
    """
        Delete unwanted __dir__ for this object.
        """
    return self._accessors | self._hidden_attrs