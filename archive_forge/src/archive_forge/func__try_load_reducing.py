from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
def _try_load_reducing(parts):
    locale = _try_load(parts)
    if locale is not None:
        return locale
    locale = _try_load(parts[:2])
    if locale is not None:
        return locale