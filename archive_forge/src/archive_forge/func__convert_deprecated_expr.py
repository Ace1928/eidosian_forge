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
def _convert_deprecated_expr(self, kwargs):
    """
        Handle old-style calculated aesthetic expression mappings

        Just converts them to use `stage` e.g.
        "stat(count)" to after_stat(count)
        "..count.." to after_stat(count)
        """
    for name, value in kwargs.items():
        if not isinstance(value, stage) and is_calculated_aes(value):
            _after_stat = strip_calculated_markers(value)
            kwargs[name] = after_stat(_after_stat)
    return kwargs