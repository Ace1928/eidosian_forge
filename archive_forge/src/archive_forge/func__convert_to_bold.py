from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def _convert_to_bold(crow: Sequence[str], ilevels: int) -> list[str]:
    """Convert elements in ``crow`` to bold."""
    return [f'\\textbf{{{x}}}' if j < ilevels and x.strip() not in ['', '{}'] else x for j, x in enumerate(crow)]