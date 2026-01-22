from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
def _handle_func_call(self, func: str, ctx: Any) -> None:
    if is_agg(func):
        self.set('has_agg_func', True)
        self.assert_support(not self._in_agg and (not self._in_window), ctx)
        self._agg_funcs.append(ctx)
        self._in_agg = True
    self.visitChildren(ctx)
    self._in_agg = False