from __future__ import annotations
import abc
from typing import TYPE_CHECKING
from pandas.errors import NumExprClobberingError
from pandas.core.computation.align import (
from pandas.core.computation.ops import (
from pandas.io.formats import printing
class PythonEngine(AbstractEngine):
    """
    Evaluate an expression in Python space.

    Mostly for testing purposes.
    """
    has_neg_frac = False

    def evaluate(self):
        return self.expr()

    def _evaluate(self) -> None:
        pass