from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
class TSQL(Dialect):
    NORMALIZATION_STRATEGY = NormalizationStrategy.CASE_INSENSITIVE
    TIME_FORMAT = "'yyyy-mm-dd hh:mm:ss'"
    SUPPORTS_SEMI_ANTI_JOIN = False
    LOG_BASE_FIRST = False
    TYPED_DIVISION = True
    CONCAT_COALESCE = True
    TIME_MAPPING = {'year': '%Y', 'dayofyear': '%j', 'day': '%d', 'dy': '%d', 'y': '%Y', 'week': '%W', 'ww': '%W', 'wk': '%W', 'hour': '%h', 'hh': '%I', 'minute': '%M', 'mi': '%M', 'n': '%M', 'second': '%S', 'ss': '%S', 's': '%-S', 'millisecond': '%f', 'ms': '%f', 'weekday': '%W', 'dw': '%W', 'month': '%m', 'mm': '%M', 'm': '%-M', 'Y': '%Y', 'YYYY': '%Y', 'YY': '%y', 'MMMM': '%B', 'MMM': '%b', 'MM': '%m', 'M': '%-m', 'dddd': '%A', 'dd': '%d', 'd': '%-d', 'HH': '%H', 'H': '%-H', 'h': '%-I', 'S': '%f', 'yyyy': '%Y', 'yy': '%y'}
    CONVERT_FORMAT_MAPPING = {'0': '%b %d %Y %-I:%M%p', '1': '%m/%d/%y', '2': '%y.%m.%d', '3': '%d/%m/%y', '4': '%d.%m.%y', '5': '%d-%m-%y', '6': '%d %b %y', '7': '%b %d, %y', '8': '%H:%M:%S', '9': '%b %d %Y %-I:%M:%S:%f%p', '10': 'mm-dd-yy', '11': 'yy/mm/dd', '12': 'yymmdd', '13': '%d %b %Y %H:%M:ss:%f', '14': '%H:%M:%S:%f', '20': '%Y-%m-%d %H:%M:%S', '21': '%Y-%m-%d %H:%M:%S.%f', '22': '%m/%d/%y %-I:%M:%S %p', '23': '%Y-%m-%d', '24': '%H:%M:%S', '25': '%Y-%m-%d %H:%M:%S.%f', '100': '%b %d %Y %-I:%M%p', '101': '%m/%d/%Y', '102': '%Y.%m.%d', '103': '%d/%m/%Y', '104': '%d.%m.%Y', '105': '%d-%m-%Y', '106': '%d %b %Y', '107': '%b %d, %Y', '108': '%H:%M:%S', '109': '%b %d %Y %-I:%M:%S:%f%p', '110': '%m-%d-%Y', '111': '%Y/%m/%d', '112': '%Y%m%d', '113': '%d %b %Y %H:%M:%S:%f', '114': '%H:%M:%S:%f', '120': '%Y-%m-%d %H:%M:%S', '121': '%Y-%m-%d %H:%M:%S.%f'}
    FORMAT_TIME_MAPPING = {'y': '%B %Y', 'd': '%m/%d/%Y', 'H': '%-H', 'h': '%-I', 's': '%Y-%m-%d %H:%M:%S', 'D': '%A,%B,%Y', 'f': '%A,%B,%Y %-I:%M %p', 'F': '%A,%B,%Y %-I:%M:%S %p', 'g': '%m/%d/%Y %-I:%M %p', 'G': '%m/%d/%Y %-I:%M:%S %p', 'M': '%B %-d', 'm': '%B %-d', 'O': '%Y-%m-%dT%H:%M:%S', 'u': '%Y-%M-%D %H:%M:%S%z', 'U': '%A, %B %D, %Y %H:%M:%S%z', 'T': '%-I:%M:%S %p', 't': '%-I:%M', 'Y': '%a %Y'}

    class Tokenizer(tokens.Tokenizer):
        IDENTIFIERS = [('[', ']'), '"']
        QUOTES = ["'", '"']
        HEX_STRINGS = [('0x', ''), ('0X', '')]
        VAR_SINGLE_TOKENS = {'@', '$', '#'}
        KEYWORDS = {**tokens.Tokenizer.KEYWORDS, 'DATETIME2': TokenType.DATETIME, 'DATETIMEOFFSET': TokenType.TIMESTAMPTZ, 'DECLARE': TokenType.COMMAND, 'EXEC': TokenType.COMMAND, 'IMAGE': TokenType.IMAGE, 'MONEY': TokenType.MONEY, 'NTEXT': TokenType.TEXT, 'PRINT': TokenType.COMMAND, 'PROC': TokenType.PROCEDURE, 'REAL': TokenType.FLOAT, 'ROWVERSION': TokenType.ROWVERSION, 'SMALLDATETIME': TokenType.DATETIME, 'SMALLMONEY': TokenType.SMALLMONEY, 'SQL_VARIANT': TokenType.VARIANT, 'TOP': TokenType.TOP, 'UNIQUEIDENTIFIER': TokenType.UNIQUEIDENTIFIER, 'UPDATE STATISTICS': TokenType.COMMAND, 'XML': TokenType.XML, 'OUTPUT': TokenType.RETURNING, 'SYSTEM_USER': TokenType.CURRENT_USER, 'FOR SYSTEM_TIME': TokenType.TIMESTAMP_SNAPSHOT, 'OPTION': TokenType.OPTION}

    class Parser(parser.Parser):
        SET_REQUIRES_ASSIGNMENT_DELIMITER = False
        LOG_DEFAULTS_TO_LN = True
        ALTER_TABLE_ADD_REQUIRED_FOR_EACH_COLUMN = False
        STRING_ALIASES = True
        NO_PAREN_IF_COMMANDS = False
        QUERY_MODIFIER_PARSERS = {**parser.Parser.QUERY_MODIFIER_PARSERS, TokenType.OPTION: lambda self: ('options', self._parse_options())}
        FUNCTIONS = {**parser.Parser.FUNCTIONS, 'CHARINDEX': lambda args: exp.StrPosition(this=seq_get(args, 1), substr=seq_get(args, 0), position=seq_get(args, 2)), 'DATEADD': build_date_delta(exp.DateAdd, unit_mapping=DATE_DELTA_INTERVAL), 'DATEDIFF': _build_date_delta(exp.DateDiff, unit_mapping=DATE_DELTA_INTERVAL), 'DATENAME': _build_formatted_time(exp.TimeToStr, full_format_mapping=True), 'DATEPART': _build_formatted_time(exp.TimeToStr), 'DATETIMEFROMPARTS': _build_datetimefromparts, 'EOMONTH': _build_eomonth, 'FORMAT': _build_format, 'GETDATE': exp.CurrentTimestamp.from_arg_list, 'HASHBYTES': _build_hashbytes, 'ISNULL': exp.Coalesce.from_arg_list, 'JSON_QUERY': parser.build_extract_json_with_path(exp.JSONExtract), 'JSON_VALUE': parser.build_extract_json_with_path(exp.JSONExtractScalar), 'LEN': _build_with_arg_as_text(exp.Length), 'LEFT': _build_with_arg_as_text(exp.Left), 'RIGHT': _build_with_arg_as_text(exp.Right), 'REPLICATE': exp.Repeat.from_arg_list, 'SQUARE': lambda args: exp.Pow(this=seq_get(args, 0), expression=exp.Literal.number(2)), 'SYSDATETIME': exp.CurrentTimestamp.from_arg_list, 'SUSER_NAME': exp.CurrentUser.from_arg_list, 'SUSER_SNAME': exp.CurrentUser.from_arg_list, 'SYSTEM_USER': exp.CurrentUser.from_arg_list, 'TIMEFROMPARTS': _build_timefromparts}
        JOIN_HINTS = {'LOOP', 'HASH', 'MERGE', 'REMOTE'}
        RETURNS_TABLE_TOKENS = parser.Parser.ID_VAR_TOKENS - {TokenType.TABLE, *parser.Parser.TYPE_TOKENS}
        STATEMENT_PARSERS = {**parser.Parser.STATEMENT_PARSERS, TokenType.END: lambda self: self._parse_command()}

        def _parse_options(self) -> t.Optional[t.List[exp.Expression]]:
            if not self._match(TokenType.OPTION):
                return None

            def _parse_option() -> t.Optional[exp.Expression]:
                option = self._parse_var_from_options(OPTIONS)
                if not option:
                    return None
                self._match(TokenType.EQ)
                return self.expression(exp.QueryOption, this=option, expression=self._parse_primary_or_var())
            return self._parse_wrapped_csv(_parse_option)

        def _parse_projections(self) -> t.List[exp.Expression]:
            """
            T-SQL supports the syntax alias = expression in the SELECT's projection list,
            so we transform all parsed Selects to convert their EQ projections into Aliases.

            See: https://learn.microsoft.com/en-us/sql/t-sql/queries/select-clause-transact-sql?view=sql-server-ver16#syntax
            """
            return [exp.alias_(projection.expression, projection.this.this, copy=False) if isinstance(projection, exp.EQ) and isinstance(projection.this, exp.Column) else projection for projection in super()._parse_projections()]

        def _parse_commit_or_rollback(self) -> exp.Commit | exp.Rollback:
            """Applies to SQL Server and Azure SQL Database
            COMMIT [ { TRAN | TRANSACTION }
                [ transaction_name | @tran_name_variable ] ]
                [ WITH ( DELAYED_DURABILITY = { OFF | ON } ) ]

            ROLLBACK { TRAN | TRANSACTION }
                [ transaction_name | @tran_name_variable
                | savepoint_name | @savepoint_variable ]
            """
            rollback = self._prev.token_type == TokenType.ROLLBACK
            self._match_texts(('TRAN', 'TRANSACTION'))
            this = self._parse_id_var()
            if rollback:
                return self.expression(exp.Rollback, this=this)
            durability = None
            if self._match_pair(TokenType.WITH, TokenType.L_PAREN):
                self._match_text_seq('DELAYED_DURABILITY')
                self._match(TokenType.EQ)
                if self._match_text_seq('OFF'):
                    durability = False
                else:
                    self._match(TokenType.ON)
                    durability = True
                self._match_r_paren()
            return self.expression(exp.Commit, this=this, durability=durability)

        def _parse_transaction(self) -> exp.Transaction | exp.Command:
            """Applies to SQL Server and Azure SQL Database
            BEGIN { TRAN | TRANSACTION }
            [ { transaction_name | @tran_name_variable }
            [ WITH MARK [ 'description' ] ]
            ]
            """
            if self._match_texts(('TRAN', 'TRANSACTION')):
                transaction = self.expression(exp.Transaction, this=self._parse_id_var())
                if self._match_text_seq('WITH', 'MARK'):
                    transaction.set('mark', self._parse_string())
                return transaction
            return self._parse_as_command(self._prev)

        def _parse_returns(self) -> exp.ReturnsProperty:
            table = self._parse_id_var(any_token=False, tokens=self.RETURNS_TABLE_TOKENS)
            returns = super()._parse_returns()
            returns.set('table', table)
            return returns

        def _parse_convert(self, strict: bool, safe: t.Optional[bool]=None) -> t.Optional[exp.Expression]:
            this = self._parse_types()
            self._match(TokenType.COMMA)
            args = [this, *self._parse_csv(self._parse_conjunction)]
            convert = exp.Convert.from_arg_list(args)
            convert.set('safe', safe)
            convert.set('strict', strict)
            return convert

        def _parse_user_defined_function(self, kind: t.Optional[TokenType]=None) -> t.Optional[exp.Expression]:
            this = super()._parse_user_defined_function(kind=kind)
            if kind == TokenType.FUNCTION or isinstance(this, exp.UserDefinedFunction) or self._match(TokenType.ALIAS, advance=False):
                return this
            expressions = self._parse_csv(self._parse_function_parameter)
            return self.expression(exp.UserDefinedFunction, this=this, expressions=expressions)

        def _parse_id_var(self, any_token: bool=True, tokens: t.Optional[t.Collection[TokenType]]=None) -> t.Optional[exp.Expression]:
            is_temporary = self._match(TokenType.HASH)
            is_global = is_temporary and self._match(TokenType.HASH)
            this = super()._parse_id_var(any_token=any_token, tokens=tokens)
            if this:
                if is_global:
                    this.set('global', True)
                elif is_temporary:
                    this.set('temporary', True)
            return this

        def _parse_create(self) -> exp.Create | exp.Command:
            create = super()._parse_create()
            if isinstance(create, exp.Create):
                table = create.this.this if isinstance(create.this, exp.Schema) else create.this
                if isinstance(table, exp.Table) and table.this.args.get('temporary'):
                    if not create.args.get('properties'):
                        create.set('properties', exp.Properties(expressions=[]))
                    create.args['properties'].append('expressions', exp.TemporaryProperty())
            return create

        def _parse_if(self) -> t.Optional[exp.Expression]:
            index = self._index
            if self._match_text_seq('OBJECT_ID'):
                self._parse_wrapped_csv(self._parse_string)
                if self._match_text_seq('IS', 'NOT', 'NULL') and self._match(TokenType.DROP):
                    return self._parse_drop(exists=True)
                self._retreat(index)
            return super()._parse_if()

        def _parse_unique(self) -> exp.UniqueColumnConstraint:
            if self._match_texts(('CLUSTERED', 'NONCLUSTERED')):
                this = self.CONSTRAINT_PARSERS[self._prev.text.upper()](self)
            else:
                this = self._parse_schema(self._parse_id_var(any_token=False))
            return self.expression(exp.UniqueColumnConstraint, this=this)

        def _parse_partition(self) -> t.Optional[exp.Partition]:
            if not self._match_text_seq('WITH', '(', 'PARTITIONS'):
                return None

            def parse_range():
                low = self._parse_bitwise()
                high = self._parse_bitwise() if self._match_text_seq('TO') else None
                return self.expression(exp.PartitionRange, this=low, expression=high) if high else low
            partition = self.expression(exp.Partition, expressions=self._parse_wrapped_csv(parse_range))
            self._match_r_paren()
            return partition

    class Generator(generator.Generator):
        LIMIT_IS_TOP = True
        QUERY_HINTS = False
        RETURNING_END = False
        NVL2_SUPPORTED = False
        ALTER_TABLE_INCLUDE_COLUMN_KEYWORD = False
        LIMIT_FETCH = 'FETCH'
        COMPUTED_COLUMN_WITH_TYPE = False
        CTE_RECURSIVE_KEYWORD_REQUIRED = False
        ENSURE_BOOLS = True
        NULL_ORDERING_SUPPORTED = None
        SUPPORTS_SINGLE_ARG_CONCAT = False
        TABLESAMPLE_SEED_KEYWORD = 'REPEATABLE'
        SUPPORTS_SELECT_INTO = True
        JSON_PATH_BRACKETED_KEY_SUPPORTED = False
        SUPPORTS_TO_NUMBER = False
        OUTER_UNION_MODIFIERS = False
        EXPRESSIONS_WITHOUT_NESTED_CTES = {exp.Delete, exp.Insert, exp.Merge, exp.Select, exp.Subquery, exp.Union, exp.Update}
        SUPPORTED_JSON_PATH_PARTS = {exp.JSONPathKey, exp.JSONPathRoot, exp.JSONPathSubscript}
        TYPE_MAPPING = {**generator.Generator.TYPE_MAPPING, exp.DataType.Type.BOOLEAN: 'BIT', exp.DataType.Type.DECIMAL: 'NUMERIC', exp.DataType.Type.DATETIME: 'DATETIME2', exp.DataType.Type.DOUBLE: 'FLOAT', exp.DataType.Type.INT: 'INTEGER', exp.DataType.Type.TEXT: 'VARCHAR(MAX)', exp.DataType.Type.TIMESTAMP: 'DATETIME2', exp.DataType.Type.TIMESTAMPTZ: 'DATETIMEOFFSET', exp.DataType.Type.VARIANT: 'SQL_VARIANT'}
        TYPE_MAPPING.pop(exp.DataType.Type.NCHAR)
        TYPE_MAPPING.pop(exp.DataType.Type.NVARCHAR)
        TRANSFORMS = {**generator.Generator.TRANSFORMS, exp.AnyValue: any_value_to_max_sql, exp.ArrayToString: rename_func('STRING_AGG'), exp.AutoIncrementColumnConstraint: lambda *_: 'IDENTITY', exp.DateAdd: date_delta_sql('DATEADD'), exp.DateDiff: date_delta_sql('DATEDIFF'), exp.CTE: transforms.preprocess([qualify_derived_table_outputs]), exp.CurrentDate: rename_func('GETDATE'), exp.CurrentTimestamp: rename_func('GETDATE'), exp.DateStrToDate: datestrtodate_sql, exp.Extract: rename_func('DATEPART'), exp.GeneratedAsIdentityColumnConstraint: generatedasidentitycolumnconstraint_sql, exp.GroupConcat: _string_agg_sql, exp.If: rename_func('IIF'), exp.JSONExtract: _json_extract_sql, exp.JSONExtractScalar: _json_extract_sql, exp.LastDay: lambda self, e: self.func('EOMONTH', e.this), exp.Max: max_or_greatest, exp.MD5: lambda self, e: self.func('HASHBYTES', exp.Literal.string('MD5'), e.this), exp.Min: min_or_least, exp.NumberToStr: _format_sql, exp.ParseJSON: lambda self, e: self.sql(e, 'this'), exp.Select: transforms.preprocess([transforms.eliminate_distinct_on, transforms.eliminate_semi_and_anti_joins, transforms.eliminate_qualify]), exp.StrPosition: lambda self, e: self.func('CHARINDEX', e.args.get('substr'), e.this, e.args.get('position')), exp.Subquery: transforms.preprocess([qualify_derived_table_outputs]), exp.SHA: lambda self, e: self.func('HASHBYTES', exp.Literal.string('SHA1'), e.this), exp.SHA2: lambda self, e: self.func('HASHBYTES', exp.Literal.string(f'SHA2_{e.args.get('length', 256)}'), e.this), exp.TemporaryProperty: lambda self, e: '', exp.TimeStrToTime: timestrtotime_sql, exp.TimeToStr: _format_sql, exp.Trim: trim_sql, exp.TsOrDsAdd: date_delta_sql('DATEADD', cast=True), exp.TsOrDsDiff: date_delta_sql('DATEDIFF')}
        TRANSFORMS.pop(exp.ReturnsProperty)
        PROPERTIES_LOCATION = {**generator.Generator.PROPERTIES_LOCATION, exp.VolatileProperty: exp.Properties.Location.UNSUPPORTED}

        def select_sql(self, expression: exp.Select) -> str:
            if expression.args.get('offset'):
                if not expression.args.get('order'):
                    expression.order_by(exp.select(exp.null()).subquery(), copy=False)
                limit = expression.args.get('limit')
                if isinstance(limit, exp.Limit):
                    limit.replace(exp.Fetch(direction='FIRST', count=limit.expression))
            return super().select_sql(expression)

        def convert_sql(self, expression: exp.Convert) -> str:
            name = 'TRY_CONVERT' if expression.args.get('safe') else 'CONVERT'
            return self.func(name, expression.this, expression.expression, expression.args.get('style'))

        def queryoption_sql(self, expression: exp.QueryOption) -> str:
            option = self.sql(expression, 'this')
            value = self.sql(expression, 'expression')
            if value:
                optional_equal_sign = '= ' if option in OPTIONS_THAT_REQUIRE_EQUAL else ''
                return f'{option} {optional_equal_sign}{value}'
            return option

        def lateral_op(self, expression: exp.Lateral) -> str:
            cross_apply = expression.args.get('cross_apply')
            if cross_apply is True:
                return 'CROSS APPLY'
            if cross_apply is False:
                return 'OUTER APPLY'
            self.unsupported('LATERAL clause is not supported.')
            return 'LATERAL'

        def timefromparts_sql(self, expression: exp.TimeFromParts) -> str:
            nano = expression.args.get('nano')
            if nano is not None:
                nano.pop()
                self.unsupported('Specifying nanoseconds is not supported in TIMEFROMPARTS.')
            if expression.args.get('fractions') is None:
                expression.set('fractions', exp.Literal.number(0))
            if expression.args.get('precision') is None:
                expression.set('precision', exp.Literal.number(0))
            return rename_func('TIMEFROMPARTS')(self, expression)

        def timestampfromparts_sql(self, expression: exp.TimestampFromParts) -> str:
            zone = expression.args.get('zone')
            if zone is not None:
                zone.pop()
                self.unsupported('Time zone is not supported in DATETIMEFROMPARTS.')
            nano = expression.args.get('nano')
            if nano is not None:
                nano.pop()
                self.unsupported('Specifying nanoseconds is not supported in DATETIMEFROMPARTS.')
            if expression.args.get('milli') is None:
                expression.set('milli', exp.Literal.number(0))
            return rename_func('DATETIMEFROMPARTS')(self, expression)

        def setitem_sql(self, expression: exp.SetItem) -> str:
            this = expression.this
            if isinstance(this, exp.EQ) and (not isinstance(this.left, exp.Parameter)):
                return f'{self.sql(this.left)} {self.sql(this.right)}'
            return super().setitem_sql(expression)

        def boolean_sql(self, expression: exp.Boolean) -> str:
            if type(expression.parent) in BIT_TYPES:
                return '1' if expression.this else '0'
            return '(1 = 1)' if expression.this else '(1 = 0)'

        def is_sql(self, expression: exp.Is) -> str:
            if isinstance(expression.expression, exp.Boolean):
                return self.binary(expression, '=')
            return self.binary(expression, 'IS')

        def createable_sql(self, expression: exp.Create, locations: t.DefaultDict) -> str:
            sql = self.sql(expression, 'this')
            properties = expression.args.get('properties')
            if sql[:1] != '#' and any((isinstance(prop, exp.TemporaryProperty) for prop in (properties.expressions if properties else []))):
                sql = f'#{sql}'
            return sql

        def create_sql(self, expression: exp.Create) -> str:
            kind = expression.kind
            exists = expression.args.pop('exists', None)
            sql = super().create_sql(expression)
            like_property = expression.find(exp.LikeProperty)
            if like_property:
                ctas_expression = like_property.this
            else:
                ctas_expression = expression.expression
            table = expression.find(exp.Table)
            if kind == 'TABLE' and ctas_expression:
                ctas_with = ctas_expression.args.get('with')
                if ctas_with:
                    ctas_with = ctas_with.pop()
                if isinstance(ctas_expression, exp.UNWRAPPED_QUERIES):
                    ctas_expression = ctas_expression.subquery()
                select_into = exp.select('*').from_(exp.alias_(ctas_expression, 'temp', table=True))
                select_into.set('into', exp.Into(this=table))
                select_into.set('with', ctas_with)
                if like_property:
                    select_into.limit(0, copy=False)
                sql = self.sql(select_into)
            if exists:
                identifier = self.sql(exp.Literal.string(exp.table_name(table) if table else ''))
                sql = self.sql(exp.Literal.string(sql))
                if kind == 'SCHEMA':
                    sql = f'IF NOT EXISTS (SELECT * FROM information_schema.schemata WHERE schema_name = {identifier}) EXEC({sql})'
                elif kind == 'TABLE':
                    assert table
                    where = exp.and_(exp.column('table_name').eq(table.name), exp.column('table_schema').eq(table.db) if table.db else None, exp.column('table_catalog').eq(table.catalog) if table.catalog else None)
                    sql = f'IF NOT EXISTS (SELECT * FROM information_schema.tables WHERE {where}) EXEC({sql})'
                elif kind == 'INDEX':
                    index = self.sql(exp.Literal.string(expression.this.text('this')))
                    sql = f'IF NOT EXISTS (SELECT * FROM sys.indexes WHERE object_id = object_id({identifier}) AND name = {index}) EXEC({sql})'
            elif expression.args.get('replace'):
                sql = sql.replace('CREATE OR REPLACE ', 'CREATE OR ALTER ', 1)
            return self.prepend_ctes(expression, sql)

        def offset_sql(self, expression: exp.Offset) -> str:
            return f'{super().offset_sql(expression)} ROWS'

        def version_sql(self, expression: exp.Version) -> str:
            name = 'SYSTEM_TIME' if expression.name == 'TIMESTAMP' else expression.name
            this = f'FOR {name}'
            expr = expression.expression
            kind = expression.text('kind')
            if kind in ('FROM', 'BETWEEN'):
                args = expr.expressions
                sep = 'TO' if kind == 'FROM' else 'AND'
                expr_sql = f'{self.sql(seq_get(args, 0))} {sep} {self.sql(seq_get(args, 1))}'
            else:
                expr_sql = self.sql(expr)
            expr_sql = f' {expr_sql}' if expr_sql else ''
            return f'{this} {kind}{expr_sql}'

        def returnsproperty_sql(self, expression: exp.ReturnsProperty) -> str:
            table = expression.args.get('table')
            table = f'{table} ' if table else ''
            return f'RETURNS {table}{self.sql(expression, 'this')}'

        def returning_sql(self, expression: exp.Returning) -> str:
            into = self.sql(expression, 'into')
            into = self.seg(f'INTO {into}') if into else ''
            return f'{self.seg('OUTPUT')} {self.expressions(expression, flat=True)}{into}'

        def transaction_sql(self, expression: exp.Transaction) -> str:
            this = self.sql(expression, 'this')
            this = f' {this}' if this else ''
            mark = self.sql(expression, 'mark')
            mark = f' WITH MARK {mark}' if mark else ''
            return f'BEGIN TRANSACTION{this}{mark}'

        def commit_sql(self, expression: exp.Commit) -> str:
            this = self.sql(expression, 'this')
            this = f' {this}' if this else ''
            durability = expression.args.get('durability')
            durability = f' WITH (DELAYED_DURABILITY = {('ON' if durability else 'OFF')})' if durability is not None else ''
            return f'COMMIT TRANSACTION{this}{durability}'

        def rollback_sql(self, expression: exp.Rollback) -> str:
            this = self.sql(expression, 'this')
            this = f' {this}' if this else ''
            return f'ROLLBACK TRANSACTION{this}'

        def identifier_sql(self, expression: exp.Identifier) -> str:
            identifier = super().identifier_sql(expression)
            if expression.args.get('global'):
                identifier = f'##{identifier}'
            elif expression.args.get('temporary'):
                identifier = f'#{identifier}'
            return identifier

        def constraint_sql(self, expression: exp.Constraint) -> str:
            this = self.sql(expression, 'this')
            expressions = self.expressions(expression, flat=True, sep=' ')
            return f'CONSTRAINT {this} {expressions}'

        def length_sql(self, expression: exp.Length) -> str:
            return self._uncast_text(expression, 'LEN')

        def right_sql(self, expression: exp.Right) -> str:
            return self._uncast_text(expression, 'RIGHT')

        def left_sql(self, expression: exp.Left) -> str:
            return self._uncast_text(expression, 'LEFT')

        def _uncast_text(self, expression: exp.Expression, name: str) -> str:
            this = expression.this
            if isinstance(this, exp.Cast) and this.is_type(exp.DataType.Type.TEXT):
                this_sql = self.sql(this, 'this')
            else:
                this_sql = self.sql(this)
            expression_sql = self.sql(expression, 'expression')
            return self.func(name, this_sql, expression_sql if expression_sql else None)

        def partition_sql(self, expression: exp.Partition) -> str:
            return f'WITH (PARTITIONS({self.expressions(expression, flat=True)}))'

        def altertable_sql(self, expression: exp.AlterTable) -> str:
            actions = expression.args['actions']
            if isinstance(actions[0], exp.RenameTable):
                table = self.sql(expression.this)
                target = actions[0].this
                target = self.sql(exp.table_(target.this) if isinstance(target, exp.Table) else target)
                return f"EXEC sp_rename '{table}', '{target}'"
            return super().altertable_sql(expression)