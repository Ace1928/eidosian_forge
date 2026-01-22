from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
def _get_numbering_system(locale: Locale, numbering_system: Literal['default'] | str='latn') -> str:
    if numbering_system == 'default':
        return locale.default_numbering_system
    else:
        return numbering_system