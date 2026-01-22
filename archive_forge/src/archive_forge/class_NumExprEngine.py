from __future__ import annotations
import abc
from typing import TYPE_CHECKING
from pandas.errors import NumExprClobberingError
from pandas.core.computation.align import (
from pandas.core.computation.ops import (
from pandas.io.formats import printing
class NumExprEngine(AbstractEngine):
    """NumExpr engine class"""
    has_neg_frac = True

    def _evaluate(self):
        import numexpr as ne
        s = self.convert()
        env = self.expr.env
        scope = env.full_scope
        _check_ne_builtin_clash(self.expr)
        return ne.evaluate(s, local_dict=scope)