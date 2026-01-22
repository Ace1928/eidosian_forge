from __future__ import annotations
import itertools
import logging
import math
from typing import Any, Callable, Dict, Iterator, List, Tuple, cast
from fontTools.designspaceLib import (
from fontTools.designspaceLib.statNames import StatNames, getStatNames
from fontTools.designspaceLib.types import (
def _conditionSetFrom(conditionSet: List[Dict[str, Any]]) -> ConditionSet:
    c: Dict[str, Range] = {}
    for condition in conditionSet:
        minimum, maximum = (condition.get('minimum'), condition.get('maximum'))
        c[condition['name']] = Range(minimum if minimum is not None else -math.inf, maximum if maximum is not None else math.inf)
    return c