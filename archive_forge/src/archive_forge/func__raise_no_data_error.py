from __future__ import annotations
import os
import pickle
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from babel import localedata
from babel.plural import PluralRule
def _raise_no_data_error():
    raise RuntimeError('The babel data files are not available. This usually happens because you are using a source checkout from Babel and you did not build the data files.  Just make sure to run "python setup.py import_cldr" before installing the library.')