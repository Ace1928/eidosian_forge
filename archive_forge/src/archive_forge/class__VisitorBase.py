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
class _VisitorBase(FugueSQLVisitor):

    def __init__(self, sql: FugueSQLParser):
        super().__init__(sql)

    def visitFugueIdentifier(self, ctx: fp.FugueIdentifierContext) -> str:
        return self.ctxToStr(ctx)

    def collectChildren(self, node: Tree, tp: Type) -> List[Any]:
        result: List[Any] = []
        n = node.getChildCount()
        for i in range(n):
            c = node.getChild(i)
            if isinstance(c, tp):
                result.append(c.accept(self))
        return result

    def ctxToStr(self, node: Union[Tree, Token, None], delimit: str=' ') -> str:
        return self.sql.get_norm_text(node, delimiter=delimit)

    def to_runtime_error(self, ctx: ParserRuleContext) -> Exception:
        msg = '\n' + self.sql.get_raw_text(ctx, add_lineno=True)
        return FugueSQLRuntimeError(msg)

    @no_type_check
    def to_kv(self, ctx: Tree) -> Tuple[Any, Any]:
        k = ctx.key.accept(self)
        v = ctx.value.accept(self)
        return (k, v)

    def get_dict(self, ctx: Tree, *keys: Any) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for k in keys:
            v = getattr(ctx, k)
            if v is not None:
                res[k] = self.visit(v)
        return res

    def visitFugueJsonObj(self, ctx: fp.FugueJsonObjContext) -> Any:
        pairs = ctx.fugueJsonPairs()
        if pairs is None:
            return dict()
        return pairs.accept(self)

    def visitFugueJsonPairs(self, ctx: fp.FugueJsonPairsContext) -> Dict:
        return dict(self.collectChildren(ctx, fp.FugueJsonPairContext))

    def visitFugueJsonPair(self, ctx: fp.FugueJsonPairContext) -> Any:
        return self.to_kv(ctx)

    def visitFugueJsonArray(self, ctx: fp.FugueJsonArrayContext) -> List[Any]:
        return self.collectChildren(ctx, fp.FugueJsonValueContext)

    def visitFugueJsonString(self, ctx: fp.FugueJsonKeyContext) -> Any:
        return eval(self.ctxToStr(ctx))

    def visitFugueJsonNumber(self, ctx: fp.FugueJsonKeyContext) -> Any:
        return eval(self.ctxToStr(ctx))

    def visitFugueJsonBool(self, ctx: fp.FugueJsonKeyContext) -> bool:
        return to_bool(self.ctxToStr(ctx))

    def visitFugueJsonNull(self, ctx: fp.FugueJsonKeyContext) -> Any:
        return None

    def visitFugueRenamePair(self, ctx: fp.FugueRenamePairContext) -> Tuple:
        return self.to_kv(ctx)

    def visitFugueRenameExpression(self, ctx: fp.FugueRenameExpressionContext) -> Dict[str, str]:
        return dict(self.collectChildren(ctx, fp.FugueRenamePairContext))

    def visitFugueWildSchema(self, ctx: fp.FugueWildSchemaContext) -> str:
        schema = ','.join(self.collectChildren(ctx, fp.FugueWildSchemaPairContext))
        ops = ''.join(self.collectChildren(ctx, fp.FugueSchemaOpContext))
        return schema + ops

    def visitFugueWildSchemaPair(self, ctx: fp.FugueWildSchemaPairContext) -> str:
        if ctx.pair is not None:
            return str(Schema([self.visit(ctx.pair)]))
        else:
            return '*'

    def visitFugueSchemaOp(self, ctx: fp.FugueSchemaOpContext) -> str:
        return self.ctxToStr(ctx, delimit='')

    def visitFugueSchema(self, ctx: fp.FugueSchemaContext) -> Schema:
        return Schema(self.collectChildren(ctx, fp.FugueSchemaPairContext))

    def visitFugueSchemaPair(self, ctx: fp.FugueSchemaPairContext) -> Any:
        tp = self.to_kv(ctx)
        return (unquote_name(tp[0]), tp[1])

    def visitFugueSchemaSimpleType(self, ctx: fp.FugueSchemaSimpleTypeContext) -> pa.DataType:
        return to_pa_datatype(self.ctxToStr(ctx))

    def visitFugueSchemaListType(self, ctx: fp.FugueSchemaListTypeContext) -> pa.DataType:
        tp = self.visit(ctx.fugueSchemaType())
        return pa.list_(tp)

    def visitFugueSchemaStructType(self, ctx: fp.FugueSchemaStructTypeContext) -> pa.DataType:
        fields = self.visit(ctx.fugueSchema()).fields
        return pa.struct(fields)

    def visitFugueSchemaMapType(self, ctx: fp.FugueSchemaMapTypeContext) -> pa.DataType:
        tps = self.collectChildren(ctx, fp.FugueSchemaTypeContext)
        return pa.map_(tps[0], tps[1])

    def visitFuguePrepartition(self, ctx: fp.FuguePrepartitionContext) -> PartitionSpec:
        params = self.get_dict(ctx, 'algo', 'num', 'by', 'presort')
        return PartitionSpec(**params)

    def visitFuguePartitionAlgo(self, ctx: fp.FuguePartitionAlgoContext) -> str:
        return self.ctxToStr(ctx).lower()

    def visitFuguePartitionNum(self, ctx: fp.FuguePartitionNumContext) -> str:
        return self.ctxToStr(ctx, delimit='').upper()

    def visitFugueCols(self, ctx: fp.FugueColsContext) -> List[str]:
        return self.collectChildren(ctx, fp.FugueColumnIdentifierContext)

    def visitFugueColsSort(self, ctx: fp.FugueColsSortContext) -> str:
        return ','.join(self.collectChildren(ctx, fp.FugueColSortContext))

    def visitFugueColSort(self, ctx: fp.FugueColSortContext) -> str:
        return self.ctxToStr(ctx)

    def visitFugueColumnIdentifier(self, ctx: fp.FugueColumnIdentifierContext) -> str:
        return unquote_name(self.ctxToStr(ctx))

    def visitFugueParamsPairs(self, ctx: fp.FugueParamsPairsContext) -> Dict:
        return dict(self.collectChildren(ctx.pairs, fp.FugueJsonPairContext))

    def visitFugueParamsObj(self, ctx: fp.FugueParamsObjContext) -> Any:
        return self.visit(ctx.obj)

    def visitFugueExtension(self, ctx: fp.FugueExtensionContext) -> Any:
        s = self.ctxToStr(ctx, delimit='')
        if ctx.domain is None:
            return s
        p = safe_split_out_of_quote(s, ':', 1)
        return (unquote_name(p[0]), p[1])

    def visitFugueSingleOutputExtensionCommon(self, ctx: fp.FugueSingleOutputExtensionCommonContext) -> Dict[str, Any]:
        return self.get_dict(ctx, 'fugueUsing', 'params', 'schema')

    def visitFugueSingleOutputExtensionCommonWild(self, ctx: fp.FugueSingleOutputExtensionCommonContext) -> Dict[str, Any]:
        return self.get_dict(ctx, 'fugueUsing', 'params', 'schema')

    def visitFugueAssignment(self, ctx: fp.FugueAssignmentContext) -> Tuple:
        varname = self.ctxToStr(ctx.varname, delimit='')
        sign = self.ctxToStr(ctx.sign, delimit='')
        return (varname, sign)

    def visitFugueSampleMethod(self, ctx: fp.FugueSampleMethodContext) -> Tuple:
        if ctx.rows is not None:
            n: Any = int(self.ctxToStr(ctx.rows))
        else:
            n = None
        if ctx.percentage is not None:
            frac: Any = float(self.ctxToStr(ctx.percentage)) / 100.0
        else:
            frac = None
        return (n, frac)

    def visitFugueZipType(self, ctx: fp.FugueZipTypeContext) -> str:
        return self.ctxToStr(ctx, delimit='_').lower()

    def visitFugueLoadColumns(self, ctx: fp.FugueLoadColumnsContext) -> Any:
        if ctx.schema is not None:
            return str(self.visit(ctx.schema))
        else:
            return self.visit(ctx.cols)

    def visitFugueSaveMode(self, ctx: fp.FugueSaveModeContext) -> str:
        mode = self.ctxToStr(ctx).lower()
        if mode == 'to':
            mode = 'error'
        return mode

    def visitFugueFileFormat(self, ctx: fp.FugueFileFormatContext) -> str:
        return self.ctxToStr(ctx).lower()

    def visitFuguePath(self, ctx: fp.FuguePathContext) -> Any:
        return eval(self.ctxToStr(ctx))

    def visitFugueCheckpointWeak(self, ctx: fp.FugueCheckpointWeakContext) -> Any:
        lazy = ctx.LAZY() is not None
        data = self.get_dict(ctx, 'params')
        if 'params' not in data:
            return lambda name, x: x.persist()
        else:
            return lambda name, x: x.weak_checkpoint(lazy=lazy, **data['params'])

    def visitFugueCheckpointStrong(self, ctx: fp.FugueCheckpointStrongContext) -> Any:
        lazy = ctx.LAZY() is not None
        data = self.get_dict(ctx, 'partition', 'single', 'params')
        return lambda name, x: x.strong_checkpoint(lazy=lazy, partition=data.get('partition'), single='single' in data, **data.get('params', {}))

    def visitFugueCheckpointDeterministic(self, ctx: fp.FugueCheckpointDeterministicContext) -> Any:

        def _func(name: str, x: WorkflowDataFrame) -> WorkflowDataFrame:
            data = self.get_dict(ctx, 'ns', 'partition', 'single', 'params')
            x.deterministic_checkpoint(lazy=ctx.LAZY() is not None, partition=data.get('partition'), single='single' in data, namespace=data.get('ns'), **data.get('params', {}))
            return x
        return _func

    def visitFugueYield(self, ctx: fp.FugueYieldContext) -> Any:

        def _func(name: str, x: WorkflowDataFrame) -> WorkflowDataFrame:
            yield_name = self.ctxToStr(ctx.name) if ctx.name is not None else name
            assert_or_throw(yield_name is not None, 'yield name is not specified')
            if ctx.DATAFRAME() is None:
                if ctx.FILE() is not None:
                    x.yield_file_as(yield_name)
                elif ctx.TABLE() is not None:
                    x.yield_table_as(yield_name)
                else:
                    raise NotImplementedError(self.ctxToStr(ctx))
            else:
                x.yield_dataframe_as(yield_name, as_local=ctx.LOCAL() is not None)
            return x
        return _func

    def visitFugueCheckpointNamespace(self, ctx: fp.FugueCheckpointNamespaceContext):
        return str(eval(self.ctxToStr(ctx)))