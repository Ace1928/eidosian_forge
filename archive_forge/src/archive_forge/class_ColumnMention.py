from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Tuple, Union
from antlr4.tree.Tree import TerminalNode, Token, Tree
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from _qpd_antlr import QPDParser as qp
from qpd._parser.sql import QPDSql
from triad.utils.schema import unquote_name
from qpd.constants import AGGREGATION_FUNCTIONS, JOIN_TYPES
from qpd.dataframe import DataFrame, DataFrames
from qpd.workflow import QPDWorkflow
class ColumnMention(object):

    def __init__(self, df_name: str, col_name: str):
        self.df_name = df_name
        self.col_name = col_name
        self.encoded = '_' + to_uuid(df_name, col_name).split('-')[-1]
        self.expr = df_name + '.' + col_name

    def __repr__(self) -> str:
        return self.expr

    def __hash__(self) -> int:
        return hash((self.df_name, self.col_name))

    def __eq__(self, other: Any) -> bool:
        return self.df_name == other.df_name and self.col_name == other.col_name