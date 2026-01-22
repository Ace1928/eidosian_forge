from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
class FromVisitor(ExpressionVisitor):

    def __init__(self, context: VisitorContext):
        super().__init__(context)
        self._encoded_map: Dict[str, str] = {}

    def visitRegularQuerySpecification(self, ctx: qp.RegularQuerySpecificationContext) -> None:
        if len(self.joins) == 0:
            self.update_current(None)
            self.set('encoded_map', self._encoded_map)
            return
        joined = self._extract_df(self.joins[0].df_name)
        for join in self.joins[1:]:
            name = join.df_name
            df = self._extract_df(name)
            left, right = (joined, df)
            left_cols: List[Column] = []
            right_cols: List[Column] = []
            joined_cols = list(joined.keys())
            if 'semi' not in join.join_type and 'anti' not in join.join_type:
                joined_cols += list(df.keys())
            on: List[str] = []
            if len(join.conditions) > 0:
                for i in range(len(join.conditions)):
                    join_col_name = f'_{i}_'
                    left_cols.append(self._join_col(join.conditions[i][0], joined).rename(join_col_name))
                    right_cols.append(self._join_col(join.conditions[i][1], df).rename(join_col_name))
                    on.append(f'_{i}_')
                left = WorkflowDataFrame(joined, *left_cols)
                right = WorkflowDataFrame(df, *right_cols)
            joined = self.workflow.op_to_df(joined_cols, 'join', left, right, on=on, join_type=join.join_type)
        self.update_current(joined)
        self.set('encoded_map', self._encoded_map)

    def _join_col(self, ctx: Any, df: WorkflowDataFrame) -> Column:
        self.update_current(df)
        return self._get_single_column(ctx)

    def _extract_df(self, name: str) -> WorkflowDataFrame:
        cols: List[Column] = []
        if self.get('select_all', False):
            df = self.dfs[name]
            for k in df.keys():
                cm = ColumnMention(name, k)
                cols.append(df[k].rename(cm.encoded))
                self._encoded_map[cm.encoded] = k
        else:
            for m in self.all_column_mentions:
                if m.df_name == name:
                    cols.append(self.dfs[name][m.col_name].rename(m.encoded))
                    self._encoded_map[m.encoded] = m.col_name
        return WorkflowDataFrame(*cols)