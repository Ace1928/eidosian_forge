import datetime
import importlib
import inspect
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import six  # type: ignore
from triad.utils.assertion import assert_or_throw
def _parse_value_and_unit(exp: Any) -> Tuple[float, str]:
    try:
        assert exp is not None
        if isinstance(exp, (int, float)):
            return (float(exp), '')
        exp = str(exp).replace(' ', '').lower()
        i = 1 if exp.startswith('-') else 0
        while i < len(exp):
            if (exp[i] < '0' or exp[i] > '9') and exp[i] != '.':
                break
            i += 1
        return (float(exp[:i]), exp[i:])
    except (ValueError, AssertionError):
        raise ValueError(f'Invalid expression {exp}')