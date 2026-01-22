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
@property
def _scaled(self) -> dict[str, Any]:
    """
        Return only the aesthetics mapped to after scaling

        The mapping is a dict of the form ``{name: expr}``, i.e the
        stage class has been peeled off.
        """
    d = {}
    for name, value in self.items():
        if isinstance(value, stage) and value.after_scale is not None:
            d[name] = value.after_scale
    return d