from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
@property
def time_zones(self) -> localedata.LocaleDataDict:
    """Locale display names for time zones.

        .. note:: The format of the value returned may change between
                  Babel versions.

        >>> Locale('en', 'US').time_zones['Europe/London']['long']['daylight']
        u'British Summer Time'
        >>> Locale('en', 'US').time_zones['America/St_Johns']['city']
        u'St. John’s'
        """
    return self._data['time_zones']