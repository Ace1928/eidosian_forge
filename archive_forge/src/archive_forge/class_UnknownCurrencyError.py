from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
class UnknownCurrencyError(Exception):
    """Exception thrown when a currency is requested for which no data is available.
    """

    def __init__(self, identifier: str) -> None:
        """Create the exception.
        :param identifier: the identifier string of the unsupported currency
        """
        Exception.__init__(self, f'Unknown currency {identifier!r}.')
        self.identifier = identifier