from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def weekend_end(self) -> int:
    """The day the weekend ends, with 0 being Monday.

        >>> Locale('de', 'DE').weekend_end
        6
        """
    return self._data['week_data']['weekend_end']