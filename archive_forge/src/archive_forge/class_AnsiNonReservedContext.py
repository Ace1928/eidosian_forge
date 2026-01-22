from antlr4 import *
from io import StringIO
import sys
class AnsiNonReservedContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser

    def ADD(self):
        return self.getToken(fugue_sqlParser.ADD, 0)

    def AFTER(self):
        return self.getToken(fugue_sqlParser.AFTER, 0)

    def ALTER(self):
        return self.getToken(fugue_sqlParser.ALTER, 0)

    def ANALYZE(self):
        return self.getToken(fugue_sqlParser.ANALYZE, 0)

    def ARCHIVE(self):
        return self.getToken(fugue_sqlParser.ARCHIVE, 0)

    def ARRAY(self):
        return self.getToken(fugue_sqlParser.ARRAY, 0)

    def ASC(self):
        return self.getToken(fugue_sqlParser.ASC, 0)

    def AT(self):
        return self.getToken(fugue_sqlParser.AT, 0)

    def BETWEEN(self):
        return self.getToken(fugue_sqlParser.BETWEEN, 0)

    def BUCKET(self):
        return self.getToken(fugue_sqlParser.BUCKET, 0)

    def BUCKETS(self):
        return self.getToken(fugue_sqlParser.BUCKETS, 0)

    def BY(self):
        return self.getToken(fugue_sqlParser.BY, 0)

    def CACHE(self):
        return self.getToken(fugue_sqlParser.CACHE, 0)

    def CASCADE(self):
        return self.getToken(fugue_sqlParser.CASCADE, 0)

    def CHANGE(self):
        return self.getToken(fugue_sqlParser.CHANGE, 0)

    def CLEAR(self):
        return self.getToken(fugue_sqlParser.CLEAR, 0)

    def CLUSTER(self):
        return self.getToken(fugue_sqlParser.CLUSTER, 0)

    def CLUSTERED(self):
        return self.getToken(fugue_sqlParser.CLUSTERED, 0)

    def CODEGEN(self):
        return self.getToken(fugue_sqlParser.CODEGEN, 0)

    def COLLECTION(self):
        return self.getToken(fugue_sqlParser.COLLECTION, 0)

    def COLUMNS(self):
        return self.getToken(fugue_sqlParser.COLUMNS, 0)

    def COMMENT(self):
        return self.getToken(fugue_sqlParser.COMMENT, 0)

    def COMMIT(self):
        return self.getToken(fugue_sqlParser.COMMIT, 0)

    def COMPACT(self):
        return self.getToken(fugue_sqlParser.COMPACT, 0)

    def COMPACTIONS(self):
        return self.getToken(fugue_sqlParser.COMPACTIONS, 0)

    def COMPUTE(self):
        return self.getToken(fugue_sqlParser.COMPUTE, 0)

    def CONCATENATE(self):
        return self.getToken(fugue_sqlParser.CONCATENATE, 0)

    def COST(self):
        return self.getToken(fugue_sqlParser.COST, 0)

    def CUBE(self):
        return self.getToken(fugue_sqlParser.CUBE, 0)

    def CURRENT(self):
        return self.getToken(fugue_sqlParser.CURRENT, 0)

    def DATA(self):
        return self.getToken(fugue_sqlParser.DATA, 0)

    def DATABASE(self):
        return self.getToken(fugue_sqlParser.DATABASE, 0)

    def DATABASES(self):
        return self.getToken(fugue_sqlParser.DATABASES, 0)

    def DBPROPERTIES(self):
        return self.getToken(fugue_sqlParser.DBPROPERTIES, 0)

    def DEFINED(self):
        return self.getToken(fugue_sqlParser.DEFINED, 0)

    def DELETE(self):
        return self.getToken(fugue_sqlParser.DELETE, 0)

    def DELIMITED(self):
        return self.getToken(fugue_sqlParser.DELIMITED, 0)

    def DESC(self):
        return self.getToken(fugue_sqlParser.DESC, 0)

    def DESCRIBE(self):
        return self.getToken(fugue_sqlParser.DESCRIBE, 0)

    def DFS(self):
        return self.getToken(fugue_sqlParser.DFS, 0)

    def DIRECTORIES(self):
        return self.getToken(fugue_sqlParser.DIRECTORIES, 0)

    def DIRECTORY(self):
        return self.getToken(fugue_sqlParser.DIRECTORY, 0)

    def DISTRIBUTE(self):
        return self.getToken(fugue_sqlParser.DISTRIBUTE, 0)

    def DIV(self):
        return self.getToken(fugue_sqlParser.DIV, 0)

    def DROP(self):
        return self.getToken(fugue_sqlParser.DROP, 0)

    def ESCAPED(self):
        return self.getToken(fugue_sqlParser.ESCAPED, 0)

    def EXCHANGE(self):
        return self.getToken(fugue_sqlParser.EXCHANGE, 0)

    def EXISTS(self):
        return self.getToken(fugue_sqlParser.EXISTS, 0)

    def EXPLAIN(self):
        return self.getToken(fugue_sqlParser.EXPLAIN, 0)

    def EXPORT(self):
        return self.getToken(fugue_sqlParser.EXPORT, 0)

    def EXTENDED(self):
        return self.getToken(fugue_sqlParser.EXTENDED, 0)

    def EXTERNAL(self):
        return self.getToken(fugue_sqlParser.EXTERNAL, 0)

    def EXTRACT(self):
        return self.getToken(fugue_sqlParser.EXTRACT, 0)

    def FIELDS(self):
        return self.getToken(fugue_sqlParser.FIELDS, 0)

    def FILEFORMAT(self):
        return self.getToken(fugue_sqlParser.FILEFORMAT, 0)

    def FIRST(self):
        return self.getToken(fugue_sqlParser.FIRST, 0)

    def FOLLOWING(self):
        return self.getToken(fugue_sqlParser.FOLLOWING, 0)

    def FORMAT(self):
        return self.getToken(fugue_sqlParser.FORMAT, 0)

    def FORMATTED(self):
        return self.getToken(fugue_sqlParser.FORMATTED, 0)

    def FUNCTION(self):
        return self.getToken(fugue_sqlParser.FUNCTION, 0)

    def FUNCTIONS(self):
        return self.getToken(fugue_sqlParser.FUNCTIONS, 0)

    def GLOBAL(self):
        return self.getToken(fugue_sqlParser.GLOBAL, 0)

    def GROUPING(self):
        return self.getToken(fugue_sqlParser.GROUPING, 0)

    def IF(self):
        return self.getToken(fugue_sqlParser.IF, 0)

    def IGNORE(self):
        return self.getToken(fugue_sqlParser.IGNORE, 0)

    def IMPORT(self):
        return self.getToken(fugue_sqlParser.IMPORT, 0)

    def INDEX(self):
        return self.getToken(fugue_sqlParser.INDEX, 0)

    def INDEXES(self):
        return self.getToken(fugue_sqlParser.INDEXES, 0)

    def INPATH(self):
        return self.getToken(fugue_sqlParser.INPATH, 0)

    def INPUTFORMAT(self):
        return self.getToken(fugue_sqlParser.INPUTFORMAT, 0)

    def INSERT(self):
        return self.getToken(fugue_sqlParser.INSERT, 0)

    def INTERVAL(self):
        return self.getToken(fugue_sqlParser.INTERVAL, 0)

    def ITEMS(self):
        return self.getToken(fugue_sqlParser.ITEMS, 0)

    def KEYS(self):
        return self.getToken(fugue_sqlParser.KEYS, 0)

    def LAST(self):
        return self.getToken(fugue_sqlParser.LAST, 0)

    def LATERAL(self):
        return self.getToken(fugue_sqlParser.LATERAL, 0)

    def LAZY(self):
        return self.getToken(fugue_sqlParser.LAZY, 0)

    def LIKE(self):
        return self.getToken(fugue_sqlParser.LIKE, 0)

    def LIMIT(self):
        return self.getToken(fugue_sqlParser.LIMIT, 0)

    def LINES(self):
        return self.getToken(fugue_sqlParser.LINES, 0)

    def LIST(self):
        return self.getToken(fugue_sqlParser.LIST, 0)

    def LOAD(self):
        return self.getToken(fugue_sqlParser.LOAD, 0)

    def LOCAL(self):
        return self.getToken(fugue_sqlParser.LOCAL, 0)

    def LOCATION(self):
        return self.getToken(fugue_sqlParser.LOCATION, 0)

    def LOCK(self):
        return self.getToken(fugue_sqlParser.LOCK, 0)

    def LOCKS(self):
        return self.getToken(fugue_sqlParser.LOCKS, 0)

    def LOGICAL(self):
        return self.getToken(fugue_sqlParser.LOGICAL, 0)

    def MACRO(self):
        return self.getToken(fugue_sqlParser.MACRO, 0)

    def MAP(self):
        return self.getToken(fugue_sqlParser.MAP, 0)

    def MATCHED(self):
        return self.getToken(fugue_sqlParser.MATCHED, 0)

    def MERGE(self):
        return self.getToken(fugue_sqlParser.MERGE, 0)

    def MSCK(self):
        return self.getToken(fugue_sqlParser.MSCK, 0)

    def NAMESPACE(self):
        return self.getToken(fugue_sqlParser.NAMESPACE, 0)

    def NAMESPACES(self):
        return self.getToken(fugue_sqlParser.NAMESPACES, 0)

    def NO(self):
        return self.getToken(fugue_sqlParser.NO, 0)

    def THENULLS(self):
        return self.getToken(fugue_sqlParser.THENULLS, 0)

    def OF(self):
        return self.getToken(fugue_sqlParser.OF, 0)

    def OPTION(self):
        return self.getToken(fugue_sqlParser.OPTION, 0)

    def OPTIONS(self):
        return self.getToken(fugue_sqlParser.OPTIONS, 0)

    def OUT(self):
        return self.getToken(fugue_sqlParser.OUT, 0)

    def OUTPUTFORMAT(self):
        return self.getToken(fugue_sqlParser.OUTPUTFORMAT, 0)

    def OVER(self):
        return self.getToken(fugue_sqlParser.OVER, 0)

    def OVERLAY(self):
        return self.getToken(fugue_sqlParser.OVERLAY, 0)

    def OVERWRITE(self):
        return self.getToken(fugue_sqlParser.OVERWRITE, 0)

    def PARTITION(self):
        return self.getToken(fugue_sqlParser.PARTITION, 0)

    def PARTITIONED(self):
        return self.getToken(fugue_sqlParser.PARTITIONED, 0)

    def PARTITIONS(self):
        return self.getToken(fugue_sqlParser.PARTITIONS, 0)

    def PERCENTLIT(self):
        return self.getToken(fugue_sqlParser.PERCENTLIT, 0)

    def PIVOT(self):
        return self.getToken(fugue_sqlParser.PIVOT, 0)

    def PLACING(self):
        return self.getToken(fugue_sqlParser.PLACING, 0)

    def POSITION(self):
        return self.getToken(fugue_sqlParser.POSITION, 0)

    def PRECEDING(self):
        return self.getToken(fugue_sqlParser.PRECEDING, 0)

    def PRINCIPALS(self):
        return self.getToken(fugue_sqlParser.PRINCIPALS, 0)

    def PROPERTIES(self):
        return self.getToken(fugue_sqlParser.PROPERTIES, 0)

    def PURGE(self):
        return self.getToken(fugue_sqlParser.PURGE, 0)

    def QUERY(self):
        return self.getToken(fugue_sqlParser.QUERY, 0)

    def RANGE(self):
        return self.getToken(fugue_sqlParser.RANGE, 0)

    def RECORDREADER(self):
        return self.getToken(fugue_sqlParser.RECORDREADER, 0)

    def RECORDWRITER(self):
        return self.getToken(fugue_sqlParser.RECORDWRITER, 0)

    def RECOVER(self):
        return self.getToken(fugue_sqlParser.RECOVER, 0)

    def REDUCE(self):
        return self.getToken(fugue_sqlParser.REDUCE, 0)

    def REFRESH(self):
        return self.getToken(fugue_sqlParser.REFRESH, 0)

    def RENAME(self):
        return self.getToken(fugue_sqlParser.RENAME, 0)

    def REPAIR(self):
        return self.getToken(fugue_sqlParser.REPAIR, 0)

    def REPLACE(self):
        return self.getToken(fugue_sqlParser.REPLACE, 0)

    def RESET(self):
        return self.getToken(fugue_sqlParser.RESET, 0)

    def RESTRICT(self):
        return self.getToken(fugue_sqlParser.RESTRICT, 0)

    def REVOKE(self):
        return self.getToken(fugue_sqlParser.REVOKE, 0)

    def RLIKE(self):
        return self.getToken(fugue_sqlParser.RLIKE, 0)

    def ROLE(self):
        return self.getToken(fugue_sqlParser.ROLE, 0)

    def ROLES(self):
        return self.getToken(fugue_sqlParser.ROLES, 0)

    def ROLLBACK(self):
        return self.getToken(fugue_sqlParser.ROLLBACK, 0)

    def ROLLUP(self):
        return self.getToken(fugue_sqlParser.ROLLUP, 0)

    def ROW(self):
        return self.getToken(fugue_sqlParser.ROW, 0)

    def ROWS(self):
        return self.getToken(fugue_sqlParser.ROWS, 0)

    def SCHEMA(self):
        return self.getToken(fugue_sqlParser.SCHEMA, 0)

    def SEPARATED(self):
        return self.getToken(fugue_sqlParser.SEPARATED, 0)

    def SERDE(self):
        return self.getToken(fugue_sqlParser.SERDE, 0)

    def SERDEPROPERTIES(self):
        return self.getToken(fugue_sqlParser.SERDEPROPERTIES, 0)

    def SET(self):
        return self.getToken(fugue_sqlParser.SET, 0)

    def SETS(self):
        return self.getToken(fugue_sqlParser.SETS, 0)

    def SHOW(self):
        return self.getToken(fugue_sqlParser.SHOW, 0)

    def SKEWED(self):
        return self.getToken(fugue_sqlParser.SKEWED, 0)

    def SORT(self):
        return self.getToken(fugue_sqlParser.SORT, 0)

    def SORTED(self):
        return self.getToken(fugue_sqlParser.SORTED, 0)

    def START(self):
        return self.getToken(fugue_sqlParser.START, 0)

    def STATISTICS(self):
        return self.getToken(fugue_sqlParser.STATISTICS, 0)

    def STORED(self):
        return self.getToken(fugue_sqlParser.STORED, 0)

    def STRATIFY(self):
        return self.getToken(fugue_sqlParser.STRATIFY, 0)

    def STRUCT(self):
        return self.getToken(fugue_sqlParser.STRUCT, 0)

    def SUBSTR(self):
        return self.getToken(fugue_sqlParser.SUBSTR, 0)

    def SUBSTRING(self):
        return self.getToken(fugue_sqlParser.SUBSTRING, 0)

    def TABLES(self):
        return self.getToken(fugue_sqlParser.TABLES, 0)

    def TABLESAMPLE(self):
        return self.getToken(fugue_sqlParser.TABLESAMPLE, 0)

    def TBLPROPERTIES(self):
        return self.getToken(fugue_sqlParser.TBLPROPERTIES, 0)

    def TEMPORARY(self):
        return self.getToken(fugue_sqlParser.TEMPORARY, 0)

    def TERMINATED(self):
        return self.getToken(fugue_sqlParser.TERMINATED, 0)

    def TOUCH(self):
        return self.getToken(fugue_sqlParser.TOUCH, 0)

    def TRANSACTION(self):
        return self.getToken(fugue_sqlParser.TRANSACTION, 0)

    def TRANSACTIONS(self):
        return self.getToken(fugue_sqlParser.TRANSACTIONS, 0)

    def TRANSFORM(self):
        return self.getToken(fugue_sqlParser.TRANSFORM, 0)

    def TRIM(self):
        return self.getToken(fugue_sqlParser.TRIM, 0)

    def TRUE(self):
        return self.getToken(fugue_sqlParser.TRUE, 0)

    def TRUNCATE(self):
        return self.getToken(fugue_sqlParser.TRUNCATE, 0)

    def UNARCHIVE(self):
        return self.getToken(fugue_sqlParser.UNARCHIVE, 0)

    def UNBOUNDED(self):
        return self.getToken(fugue_sqlParser.UNBOUNDED, 0)

    def UNCACHE(self):
        return self.getToken(fugue_sqlParser.UNCACHE, 0)

    def UNLOCK(self):
        return self.getToken(fugue_sqlParser.UNLOCK, 0)

    def UNSET(self):
        return self.getToken(fugue_sqlParser.UNSET, 0)

    def UPDATE(self):
        return self.getToken(fugue_sqlParser.UPDATE, 0)

    def USE(self):
        return self.getToken(fugue_sqlParser.USE, 0)

    def VALUES(self):
        return self.getToken(fugue_sqlParser.VALUES, 0)

    def VIEW(self):
        return self.getToken(fugue_sqlParser.VIEW, 0)

    def VIEWS(self):
        return self.getToken(fugue_sqlParser.VIEWS, 0)

    def WINDOW(self):
        return self.getToken(fugue_sqlParser.WINDOW, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_ansiNonReserved

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitAnsiNonReserved'):
            return visitor.visitAnsiNonReserved(self)
        else:
            return visitor.visitChildren(self)