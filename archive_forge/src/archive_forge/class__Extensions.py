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
class _Extensions(_VisitorBase):

    def __init__(self, sql: FugueSQLParser, hooks: FugueSQLHooks, workflow: FugueWorkflow, dialect: str, variables: Optional[Dict[str, Tuple[WorkflowDataFrame, WorkflowDataFrames, LazyWorkflowDataFrame]]]=None, last: Optional[WorkflowDataFrame]=None, global_vars: Optional[Dict[str, Any]]=None, local_vars: Optional[Dict[str, Any]]=None):
        super().__init__(sql)
        self._workflow = workflow
        self._variables: Dict[str, Tuple[WorkflowDataFrame, WorkflowDataFrames, LazyWorkflowDataFrame]] = {}
        if variables is not None:
            self._variables.update(variables)
        self._last: Optional[WorkflowDataFrame] = last
        self._hooks = hooks
        self._global_vars, self._local_vars = get_caller_global_local_vars(global_vars, local_vars)
        self._dialect = dialect

    @property
    def workflow(self) -> FugueWorkflow:
        return self._workflow

    @property
    def hooks(self) -> FugueSQLHooks:
        return self._hooks

    @property
    def variables(self) -> Dict[str, Tuple[WorkflowDataFrame, WorkflowDataFrames, LazyWorkflowDataFrame]]:
        return self._variables

    @property
    def last(self) -> WorkflowDataFrame:
        if self._last is None:
            raise FugueSQLError('latest dataframe does not exist')
        return self._last

    @property
    def global_vars(self) -> Dict[str, Any]:
        return self._global_vars

    @property
    def local_vars(self) -> Dict[str, Any]:
        return self._local_vars

    def get_df(self, key: str, ctx: fp.FugueDataFrameMemberContext) -> WorkflowDataFrame:
        assert_or_throw(key in self.variables, lambda: FugueSQLSyntaxError(f'{key} is not defined'))
        if isinstance(self.variables[key], LazyWorkflowDataFrame):
            assert_or_throw(ctx is None, FugueSQLSyntaxError("can't specify index or key for dataframe"))
            return self.variables[key].get_df()
        if isinstance(self.variables[key], WorkflowDataFrame):
            assert_or_throw(ctx is None, FugueSQLSyntaxError("can't specify index or key for dataframe"))
            return self.variables[key]
        assert_or_throw(ctx is not None, FugueSQLSyntaxError('must specify index or key for dataframes'))
        if ctx.index is not None:
            return self.variables[key][int(self.ctxToStr(ctx.index))]
        else:
            return self.variables[key][self.ctxToStr(ctx.key)]

    def visitFugueDataFrameSource(self, ctx: fp.FugueDataFrameSourceContext) -> WorkflowDataFrame:
        name = self.ctxToStr(ctx.fugueIdentifier(), delimit='')
        return self.get_df(name, ctx.fugueDataFrameMember())

    def visitFugueDataFrameNested(self, ctx: fp.FugueDataFrameNestedContext) -> WorkflowDataFrame:
        sub = _Extensions(self.sql, self.hooks, workflow=self.workflow, dialect=self._dialect, variables=self.variables, last=self._last, global_vars=self.global_vars, local_vars=self.local_vars)
        sub.visit(ctx.task)
        return sub.last

    def visitFugueDataFramePair(self, ctx: fp.FugueDataFramePairContext) -> Any:
        return self.to_kv(ctx)

    def visitFugueDataFramesList(self, ctx: fp.FugueDataFramesListContext) -> WorkflowDataFrames:
        dfs = self.collectChildren(ctx, fp.FugueDataFrameContext)
        return WorkflowDataFrames(dfs)

    def visitFugueDataFramesDict(self, ctx: fp.FugueDataFramesDictContext) -> WorkflowDataFrames:
        dfs = self.collectChildren(ctx, fp.FugueDataFramePairContext)
        return WorkflowDataFrames(dfs)

    def visitFugueTransformTask(self, ctx: fp.FugueTransformTaskContext) -> WorkflowDataFrame:
        data = self.get_dict(ctx, 'partition', 'dfs', 'params', 'callback')
        if 'dfs' not in data:
            data['dfs'] = WorkflowDataFrames(self.last)
        p = data['params']
        using = _to_transformer(p['fugueUsing'], schema=p.get('schema'), global_vars=self.global_vars, local_vars=self.local_vars)
        __modified_exception__ = self.to_runtime_error(ctx)
        return self.workflow.transform(data['dfs'], using=using, params=p.get('params'), pre_partition=data.get('partition'), callback=to_function(data['callback'], self.global_vars, self.local_vars) if 'callback' in data else None)

    def visitFugueOutputTransformTask(self, ctx: fp.FugueOutputTransformTaskContext) -> None:
        data = self.get_dict(ctx, 'partition', 'dfs', 'fugueUsing', 'params', 'callback')
        if 'dfs' not in data:
            data['dfs'] = WorkflowDataFrames(self.last)
        using = _to_output_transformer(data['fugueUsing'], global_vars=self.global_vars, local_vars=self.local_vars)
        __modified_exception__ = self.to_runtime_error(ctx)
        self.workflow.out_transform(data['dfs'], using=using, params=data.get('params'), pre_partition=data.get('partition'), callback=to_function(data['callback'], self.global_vars, self.local_vars) if 'callback' in data else None)

    def visitFugueProcessTask(self, ctx: fp.FugueProcessTaskContext) -> WorkflowDataFrame:
        data = self.get_dict(ctx, 'partition', 'dfs', 'params')
        if 'dfs' not in data:
            data['dfs'] = WorkflowDataFrames(self.last)
        p = data['params']
        using = _to_processor(p['fugueUsing'], schema=p.get('schema'), global_vars=self.global_vars, local_vars=self.local_vars)
        __modified_exception__ = self.to_runtime_error(ctx)
        return self.workflow.process(data['dfs'], using=using, params=p.get('params'), pre_partition=data.get('partition'))

    def visitFugueCreateTask(self, ctx: fp.FugueCreateTaskContext) -> WorkflowDataFrame:
        data = self.get_dict(ctx, 'params')
        p = data['params']
        using = _to_creator(p['fugueUsing'], schema=p.get('schema'), global_vars=self.global_vars, local_vars=self.local_vars)
        __modified_exception__ = self.to_runtime_error(ctx)
        return self.workflow.create(using=using, params=p.get('params'))

    def visitFugueCreateDataTask(self, ctx: fp.FugueCreateDataTaskContext) -> WorkflowDataFrame:
        data = self.get_dict(ctx, 'data', 'schema')
        __modified_exception__ = self.to_runtime_error(ctx)
        return self.workflow.df(data['data'], schema=data['schema'], data_determiner=to_uuid)

    def visitFugueZipTask(self, ctx: fp.FugueZipTaskContext) -> WorkflowDataFrame:
        data = self.get_dict(ctx, 'dfs', 'how')
        partition_spec = PartitionSpec(**self.get_dict(ctx, 'by', 'presort'))
        __modified_exception__ = self.to_runtime_error(ctx)
        return self.workflow.zip(data['dfs'], how=data.get('how', 'inner'), partition=partition_spec)

    def visitFugueOutputTask(self, ctx: fp.FugueOutputTaskContext):
        data = self.get_dict(ctx, 'dfs', 'fugueUsing', 'params', 'partition')
        if 'dfs' not in data:
            data['dfs'] = WorkflowDataFrames(self.last)
        using = _to_outputter(data['fugueUsing'], global_vars=self.global_vars, local_vars=self.local_vars)
        __modified_exception__ = self.to_runtime_error(ctx)
        self.workflow.output(data['dfs'], using=using, params=data.get('params'), pre_partition=data.get('partition'))

    def visitFuguePrintTask(self, ctx: fp.FuguePrintTaskContext) -> None:
        data = self.get_dict(ctx, 'dfs')
        if 'dfs' not in data:
            data['dfs'] = WorkflowDataFrames(self.last)
        params: Dict[str, Any] = {}
        if ctx.rows is not None:
            params['n'] = int(self.ctxToStr(ctx.rows))
        if ctx.count is not None:
            params['with_count'] = True
        if ctx.title is not None:
            params['title'] = eval(self.ctxToStr(ctx.title))
        __modified_exception__ = self.to_runtime_error(ctx)
        self.workflow.show(data['dfs'], **params)

    def visitFugueSaveTask(self, ctx: fp.FugueSaveTaskContext):
        data = self.get_dict(ctx, 'partition', 'df', 'm', 'single', 'fmt', 'path', 'params')
        if 'df' in data:
            df = data['df']
        else:
            df = self.last
        df.save(path=data['path'], fmt=data.get('fmt', ''), mode=data['m'], partition=data.get('partition'), single='single' in data, **data.get('params', {}))

    def visitFugueSaveAndUseTask(self, ctx: fp.FugueSaveAndUseTaskContext):
        data = self.get_dict(ctx, 'partition', 'df', 'm', 'single', 'fmt', 'path', 'params')
        if 'df' in data:
            df = data['df']
        else:
            df = self.last
        return df.save_and_use(path=data['path'], fmt=data.get('fmt', ''), mode=data['m'], partition=data.get('partition'), single='single' in data, **data.get('params', {}))

    def visitFugueRenameColumnsTask(self, ctx: fp.FugueRenameColumnsTaskContext):
        data = self.get_dict(ctx, 'cols', 'df')
        if 'df' in data:
            df = data['df']
        else:
            df = self.last
        return df.rename(data['cols'])

    def visitFugueAlterColumnsTask(self, ctx: fp.FugueAlterColumnsTaskContext):
        data = self.get_dict(ctx, 'cols', 'df')
        if 'df' in data:
            df = data['df']
        else:
            df = self.last
        return df.alter_columns(data['cols'])

    def visitFugueDropColumnsTask(self, ctx: fp.FugueDropColumnsTaskContext):
        data = self.get_dict(ctx, 'cols', 'df')
        if 'df' in data:
            df = data['df']
        else:
            df = self.last
        return df.drop(columns=data['cols'], if_exists=ctx.IF() is not None)

    def visitFugueDropnaTask(self, ctx: fp.FugueDropnaTaskContext):
        data = self.get_dict(ctx, 'cols', 'df')
        if 'df' in data:
            df = data['df']
        else:
            df = self.last
        params: Dict[str, Any] = {}
        params['how'] = 'any' if ctx.ANY() is not None else 'all'
        if 'cols' in data:
            params['subset'] = data['cols']
        return df.dropna(**params)

    def visitFugueFillnaTask(self, ctx: fp.FugueFillnaTaskContext):
        data = self.get_dict(ctx, 'params', 'df')
        if 'df' in data:
            df = data['df']
        else:
            df = self.last
        return df.fillna(value=data['params'])

    def visitFugueSampleTask(self, ctx: fp.FugueSampleTaskContext):
        data = self.get_dict(ctx, 'df')
        if 'df' in data:
            df = data['df']
        else:
            df = self.last
        params: Dict[str, Any] = {}
        params['replace'] = ctx.REPLACE() is not None
        if ctx.seed is not None:
            params['seed'] = int(self.ctxToStr(ctx.seed))
        n, frac = self.visit(ctx.method)
        if n is not None:
            params['n'] = n
        if frac is not None:
            params['frac'] = frac
        return df.sample(**params)

    def visitFugueTakeTask(self, ctx: fp.FugueTakeTaskContext):
        data = self.get_dict(ctx, 'partition', 'presort', 'df')
        if 'df' in data:
            df = data['df']
        else:
            df = self.last
        params: Dict[str, Any] = {}
        params['n'] = int(self.ctxToStr(ctx.rows)) or 20
        params['na_position'] = 'first' if ctx.FIRST() is not None else 'last'
        if data.get('partition'):
            _partition_spec = PartitionSpec(data.get('partition'))
            return df.partition(by=_partition_spec.partition_by, presort=_partition_spec.presort).take(**params)
        else:
            if data.get('presort'):
                params['presort'] = data.get('presort')
            return df.take(**params)

    def visitFugueLoadTask(self, ctx: fp.FugueLoadTaskContext) -> WorkflowDataFrame:
        data = self.get_dict(ctx, 'fmt', 'path', 'params', 'columns')
        __modified_exception__ = self.to_runtime_error(ctx)
        return self.workflow.load(path=data['path'], fmt=data.get('fmt', ''), columns=data.get('columns'), **data.get('params', {}))

    def visitFugueNestableTask(self, ctx: fp.FugueNestableTaskContext) -> None:
        data = self.get_dict(ctx, 'q')
        statements = list(self._beautify_sql(data['q']))
        if len(statements) == 1 and isinstance(statements[0], WorkflowDataFrame):
            df: Any = statements[0]
        else:
            __modified_exception__ = self.to_runtime_error(ctx)
            df = self.workflow.select(*statements, dialect=self._dialect)
        self._process_assignable(df, ctx)

    def visitFugueModuleTask(self, ctx: fp.FugueModuleTaskContext) -> None:
        data = self.get_dict(ctx, 'assign', 'dfs', 'fugueUsing', 'params')
        sub = _to_module(data['fugueUsing'], global_vars=self.global_vars, local_vars=self.local_vars)
        varname = data['assign'][0] if 'assign' in data else None
        if varname is not None:
            assert_or_throw(sub.has_single_output or sub.has_multiple_output, FugueSQLSyntaxError('invalid assignment for module without output'))
        if sub.has_input:
            dfs = data['dfs'] if 'dfs' in data else WorkflowDataFrames(self.last)
        else:
            dfs = WorkflowDataFrames()
        p = data['params'] if 'params' in data else {}
        if sub.has_dfs_input:
            result = sub(dfs, **p)
        elif len(dfs) == 0:
            result = sub(self.workflow, **p)
        elif len(dfs) == 1 or not dfs.has_key:
            result = sub(*list(dfs.values()), **p)
        else:
            result = sub(**dfs, **p)
        if sub.has_single_output or sub.has_multiple_output:
            self.variables[varname] = result
        if sub.has_single_output:
            self._last = result

    def visitFugueSqlEngine(self, ctx: fp.FugueSqlEngineContext) -> Tuple[Any, Dict[str, Any]]:
        data = self.get_dict(ctx, 'fugueUsing', 'params')
        try:
            engine: Any = to_type(data['fugueUsing'], SQLEngine, global_vars=self.global_vars, local_vars=self.local_vars)
        except TypeError:
            engine = str(data['fugueUsing'])
        return (engine, data.get('params', {}))

    def visitQuery(self, ctx: fp.QueryContext) -> Iterable[Any]:

        def get_sql() -> str:
            return ' '.join(['' if ctx.ctes() is None else self.ctxToStr(ctx.ctes()), self.ctxToStr(ctx.queryTerm()), self.ctxToStr(ctx.queryOrganization())]).strip()
        if ctx.fugueSqlEngine() is not None:
            engine, engine_params = self.visitFugueSqlEngine(ctx.fugueSqlEngine())
            __modified_exception__ = self.to_runtime_error(ctx)
            yield self.workflow.select(get_sql(), sql_engine=engine, sql_engine_params=engine_params, dialect=self._dialect)
        elif ctx.ctes() is None:
            yield from self._get_query_elements(ctx)
        else:
            __modified_exception__ = self.to_runtime_error(ctx)
            yield self.workflow.select(get_sql(), dialect=self._dialect)

    def visitOptionalFromClause(self, ctx: fp.OptionalFromClauseContext) -> Iterable[Any]:
        c = ctx.fromClause()
        if c is None:
            yield 'FROM'
            yield self.last
        else:
            yield from self._get_query_elements(ctx)

    def visitTableName(self, ctx: fp.TableNameContext) -> Iterable[Any]:
        table_name = self.ctxToStr(ctx.multipartIdentifier(), delimit='')
        if table_name not in self.variables:
            assert_or_throw(ctx.fugueDataFrameMember() is None, FugueSQLSyntaxError("can't specify index or key for dataframe"))
            table: Any = self.hooks.on_select_source_not_found(self.workflow, table_name)
        else:
            table = self.get_df(table_name, ctx.fugueDataFrameMember())
        if isinstance(table, str):
            yield table
            yield from self._get_query_elements(ctx.sample())
            yield from self._get_query_elements(ctx.tableAlias())
        else:
            yield table
            yield from self._get_query_elements(ctx.sample())
            if ctx.tableAlias().strictIdentifier() is not None:
                yield from self._get_query_elements(ctx.tableAlias())
            elif validate_triad_var_name(table_name):
                yield 'AS'
                yield table_name

    def visitFugueNestableTaskCollectionNoSelect(self, ctx: fp.FugueNestableTaskCollectionNoSelectContext) -> Iterable[Any]:
        last = self._last
        for i in range(ctx.getChildCount()):
            n = ctx.getChild(i)
            sub = _Extensions(self.sql, self.hooks, workflow=self.workflow, dialect=self._dialect, variables=self.variables, last=last, global_vars=self.global_vars, local_vars=self.local_vars)
            yield sub.visit(n)

    def visitSetOperation(self, ctx: fp.SetOperationContext) -> Iterable[Any]:

        def get_sub(_ctx: Tree) -> List[Any]:
            sub = list(self.visitFugueTerm(_ctx) if isinstance(_ctx, fp.FugueTermContext) else self._get_query_elements(_ctx))
            if len(sub) == 1 and isinstance(sub[0], WorkflowDataFrame):
                return ['SELECT * FROM', sub[0]]
            else:
                return sub
        yield from get_sub(ctx.left)
        yield from self._get_query_elements(ctx.theOperator)
        if ctx.setQuantifier() is not None:
            yield from self._get_query_elements(ctx.setQuantifier())
        yield from get_sub(ctx.right)

    def visitAliasedQuery(self, ctx: fp.AliasedQueryContext) -> Iterable[Any]:
        sub = list(self._get_query_elements(ctx.query()))
        if len(sub) == 1 and isinstance(sub[0], WorkflowDataFrame):
            yield sub[0]
        else:
            yield '('
            yield from sub
            yield ')'
        yield from self._get_query_elements(ctx.sample())
        yield from self._get_query_elements(ctx.tableAlias())

    def _beautify_sql(self, statements: Iterable[Any]) -> Iterable[Any]:
        current = ''
        for s in statements:
            if not isinstance(s, str):
                if current != '':
                    yield current
                yield s
                current = ''
            else:
                s = s.strip()
                if s != '':
                    if current == '' or current.endswith('.') or s.startswith('.') or current.endswith('(') or s.startswith(')'):
                        current += s
                    else:
                        current += ' ' + s
        if current != '':
            yield current

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

    def _process_assignable(self, df: WorkflowDataFrame, ctx: Tree):
        data = self.get_dict(ctx, 'assign', 'checkpoint', 'broadcast', 'y')
        if 'assign' in data:
            varname, _ = data['assign']
        else:
            varname = None
        if 'checkpoint' in data:
            data['checkpoint'](varname, df)
        if 'broadcast' in data:
            df = df.broadcast()
        if 'y' in data:
            data['y'](varname, df)
        if varname is not None:
            self.variables[varname] = df
        self._last = df