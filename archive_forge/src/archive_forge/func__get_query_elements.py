from typing import (
import pyarrow as pa
from antlr4 import ParserRuleContext
from antlr4.Token import CommonToken
from antlr4.tree.Tree import TerminalNode, Token, Tree
from fugue_sql_antlr import FugueSQLParser, FugueSQLVisitor
from fugue_sql_antlr._parser.fugue_sqlParser import fugue_sqlParser as fp
from triad import to_uuid
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import (
from triad.utils.pyarrow import to_pa_datatype
from triad.utils.schema import unquote_name, safe_split_out_of_quote
from triad.utils.string import validate_triad_var_name
from ..collections.partition import PartitionSpec
from ..exceptions import FugueSQLError, FugueSQLRuntimeError, FugueSQLSyntaxError
from ..execution.execution_engine import SQLEngine
from ..extensions.creator.convert import _to_creator
from ..extensions.outputter.convert import _to_outputter
from ..extensions.processor.convert import _to_processor
from ..extensions.transformer.convert import _to_output_transformer, _to_transformer
from ..workflow.module import _to_module
from ..workflow.workflow import FugueWorkflow, WorkflowDataFrame, WorkflowDataFrames
from ._utils import LazyWorkflowDataFrame
def _get_query_elements(self, node: Tree) -> Iterable[Any]:
    if node is None:
        return
    if isinstance(node, CommonToken):
        yield self.sql.code[node.start:node.stop + 1]
        return
    if isinstance(node, TerminalNode):
        token = node.getSymbol()
        yield self.sql.code[token.start:token.stop + 1]
    for i in range(node.getChildCount()):
        n = node.getChild(i)
        if isinstance(n, fp.TableNameContext):
            yield from self.visitTableName(n)
        elif isinstance(n, fp.OptionalFromClauseContext):
            yield from self.visitOptionalFromClause(n)
        elif isinstance(n, fp.FugueTermContext):
            yield from self.visitFugueTerm(n)
        elif isinstance(n, fp.AliasedQueryContext):
            yield from self.visitAliasedQuery(n)
        elif isinstance(n, fp.SetOperationContext):
            yield from self.visitSetOperation(n)
        else:
            yield from self._get_query_elements(n)