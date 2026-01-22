from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def _escape_symbols(row: Sequence[str]) -> list[str]:
    """Carry out string replacements for special symbols.

    Parameters
    ----------
    row : list
        List of string, that may contain special symbols.

    Returns
    -------
    list
        list of strings with the special symbols replaced.
    """
    return [x.replace('\\', '\\textbackslash ').replace('_', '\\_').replace('%', '\\%').replace('$', '\\$').replace('#', '\\#').replace('{', '\\{').replace('}', '\\}').replace('~', '\\textasciitilde ').replace('^', '\\textasciicircum ').replace('&', '\\&') if x and x != '{}' else '{}' for x in row]