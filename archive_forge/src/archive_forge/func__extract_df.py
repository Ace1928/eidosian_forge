from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
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