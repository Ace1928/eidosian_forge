from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
class NumberFormatError(ValueError):
    """Exception raised when a string cannot be parsed into a number."""

    def __init__(self, message: str, suggestions: list[str] | None=None) -> None:
        super().__init__(message)
        self.suggestions = suggestions