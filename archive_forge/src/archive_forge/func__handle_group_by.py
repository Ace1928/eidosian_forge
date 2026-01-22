from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
def _handle_group_by(self) -> None:
    for i_ctx in self.get('group_by_expressions'):
        internal_expr, _ = self._get_internal_col(i_ctx)
        self._gp_keys.append(internal_expr)