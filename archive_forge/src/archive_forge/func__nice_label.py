from __future__ import annotations
import re
import typing
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import fields
from typing import Any, Dict
import pandas as pd
from ..iapi import labels_view
from .evaluation import after_stat, stage
def _nice_label(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    elif isinstance(value, pd.Series):
        return value.name
    elif not isinstance(value, Iterable):
        return str(value)
    elif isinstance(value, Sequence) and len(value) == 1:
        return str(value[0])
    else:
        return None