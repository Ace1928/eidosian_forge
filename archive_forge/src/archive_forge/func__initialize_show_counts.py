from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def _initialize_show_counts(self, show_counts: bool | None) -> bool:
    if show_counts is None:
        return True
    else:
        return show_counts