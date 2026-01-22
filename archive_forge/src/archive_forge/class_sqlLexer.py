from antlr4 import *
from io import StringIO
import sys
class sqlLexer(Lexer):
    atn = ATNDeserializer().deserialize(serializedATN())
    decisionsToDFA = [DFA(ds, i) for i, ds in enumerate(atn.decisionToState)]
    T__0 = 1
    T__1 = 2
    T__2 = 3
    T__3 = 4
    T__4 = 5
    T__5 = 6
    T__6 = 7
    T__7 = 8
    T__8 = 9
    T__9 = 10
    T__10 = 11
    ADD = 12
    AFTER = 13
    ALL = 14
    ALTER = 15
    ANALYZE = 16
    AND = 17
    ANTI = 18
    ANY = 19
    ARCHIVE = 20
    ARRAY = 21
    AS = 22
    ASC = 23
    AT = 24
    AUTHORIZATION = 25
    BETWEEN = 26
    BOTH = 27
    BUCKET = 28
    BUCKETS = 29
    BY = 30
    CACHE = 31
    CASCADE = 32
    CASE = 33
    CAST = 34
    CHANGE = 35
    CHECK = 36
    CLEAR = 37
    CLUSTER = 38
    CLUSTERED = 39
    CODEGEN = 40
    COLLATE = 41
    COLLECTION = 42
    COLUMN = 43
    COLUMNS = 44
    COMMENT = 45
    COMMIT = 46
    COMPACT = 47
    COMPACTIONS = 48
    COMPUTE = 49
    CONCATENATE = 50
    CONSTRAINT = 51
    COST = 52
    CREATE = 53
    CROSS = 54
    CUBE = 55
    CURRENT = 56
    CURRENT_DATE = 57
    CURRENT_TIME = 58
    CURRENT_TIMESTAMP = 59
    CURRENT_USER = 60
    DATA = 61
    DATABASE = 62
    DATABASES = 63
    DAY = 64
    DBPROPERTIES = 65
    DEFINED = 66
    DELETE = 67
    DELIMITED = 68
    DESC = 69
    DESCRIBE = 70
    DFS = 71
    DIRECTORIES = 72
    DIRECTORY = 73
    DISTINCT = 74
    DISTRIBUTE = 75
    DROP = 76
    ELSE = 77
    END = 78
    ESCAPE = 79
    ESCAPED = 80
    EXCEPT = 81
    EXCHANGE = 82
    EXISTS = 83
    EXPLAIN = 84
    EXPORT = 85
    EXTENDED = 86
    EXTERNAL = 87
    EXTRACT = 88
    FALSE = 89
    FETCH = 90
    FIELDS = 91
    FILTER = 92
    FILEFORMAT = 93
    FIRST = 94
    FOLLOWING = 95
    FOR = 96
    FOREIGN = 97
    FORMAT = 98
    FORMATTED = 99
    FROM = 100
    FULL = 101
    FUNCTION = 102
    FUNCTIONS = 103
    GLOBAL = 104
    GRANT = 105
    GROUP = 106
    GROUPING = 107
    HAVING = 108
    HOUR = 109
    IF = 110
    IGNORE = 111
    IMPORT = 112
    IN = 113
    INDEX = 114
    INDEXES = 115
    INNER = 116
    INPATH = 117
    INPUTFORMAT = 118
    INSERT = 119
    INTERSECT = 120
    INTERVAL = 121
    INTO = 122
    IS = 123
    ITEMS = 124
    JOIN = 125
    KEYS = 126
    LAST = 127
    LATERAL = 128
    LAZY = 129
    LEADING = 130
    LEFT = 131
    LIKE = 132
    LIMIT = 133
    LINES = 134
    LIST = 135
    LOAD = 136
    LOCAL = 137
    LOCATION = 138
    LOCK = 139
    LOCKS = 140
    LOGICAL = 141
    MACRO = 142
    MAP = 143
    MATCHED = 144
    MERGE = 145
    MINUTE = 146
    MONTH = 147
    MSCK = 148
    NAMESPACE = 149
    NAMESPACES = 150
    NATURAL = 151
    NO = 152
    NOT = 153
    NULL = 154
    NULLS = 155
    OF = 156
    ON = 157
    ONLY = 158
    OPTION = 159
    OPTIONS = 160
    OR = 161
    ORDER = 162
    OUT = 163
    OUTER = 164
    OUTPUTFORMAT = 165
    OVER = 166
    OVERLAPS = 167
    OVERLAY = 168
    OVERWRITE = 169
    PARTITION = 170
    PARTITIONED = 171
    PARTITIONS = 172
    PERCENTLIT = 173
    PIVOT = 174
    PLACING = 175
    POSITION = 176
    PRECEDING = 177
    PRIMARY = 178
    PRINCIPALS = 179
    PROPERTIES = 180
    PURGE = 181
    QUERY = 182
    RANGE = 183
    RECORDREADER = 184
    RECORDWRITER = 185
    RECOVER = 186
    REDUCE = 187
    REFERENCES = 188
    REFRESH = 189
    RENAME = 190
    REPAIR = 191
    REPLACE = 192
    RESET = 193
    RESTRICT = 194
    REVOKE = 195
    RIGHT = 196
    RLIKE = 197
    ROLE = 198
    ROLES = 199
    ROLLBACK = 200
    ROLLUP = 201
    ROW = 202
    ROWS = 203
    SCHEMA = 204
    SECOND = 205
    SELECT = 206
    SEMI = 207
    SEPARATED = 208
    SERDE = 209
    SERDEPROPERTIES = 210
    SESSION_USER = 211
    SET = 212
    SETMINUS = 213
    SETS = 214
    SHOW = 215
    SKEWED = 216
    SOME = 217
    SORT = 218
    SORTED = 219
    START = 220
    STATISTICS = 221
    STORED = 222
    STRATIFY = 223
    STRUCT = 224
    SUBSTR = 225
    SUBSTRING = 226
    TABLE = 227
    TABLES = 228
    TABLESAMPLE = 229
    TBLPROPERTIES = 230
    TEMPORARY = 231
    TERMINATED = 232
    THEN = 233
    TO = 234
    TOUCH = 235
    TRAILING = 236
    TRANSACTION = 237
    TRANSACTIONS = 238
    TRANSFORM = 239
    TRIM = 240
    TRUE = 241
    TRUNCATE = 242
    TYPE = 243
    UNARCHIVE = 244
    UNBOUNDED = 245
    UNCACHE = 246
    UNION = 247
    UNIQUE = 248
    UNKNOWN = 249
    UNLOCK = 250
    UNSET = 251
    UPDATE = 252
    USE = 253
    USER = 254
    USING = 255
    VALUES = 256
    VIEW = 257
    VIEWS = 258
    WHEN = 259
    WHERE = 260
    WINDOW = 261
    WITH = 262
    YEAR = 263
    EQ = 264
    NSEQ = 265
    NEQ = 266
    NEQJ = 267
    LT = 268
    LTE = 269
    GT = 270
    GTE = 271
    PLUS = 272
    MINUS = 273
    ASTERISK = 274
    SLASH = 275
    PERCENT = 276
    DIV = 277
    TILDE = 278
    AMPERSAND = 279
    PIPE = 280
    CONCAT_PIPE = 281
    HAT = 282
    STRING = 283
    BIGINT_LITERAL = 284
    SMALLINT_LITERAL = 285
    TINYINT_LITERAL = 286
    INTEGER_VALUE = 287
    EXPONENT_VALUE = 288
    DECIMAL_VALUE = 289
    DOUBLE_LITERAL = 290
    BIGDECIMAL_LITERAL = 291
    IDENTIFIER = 292
    BACKQUOTED_IDENTIFIER = 293
    SIMPLE_COMMENT = 294
    BRACKETED_COMMENT = 295
    WS = 296
    UNRECOGNIZED = 297
    channelNames = [u'DEFAULT_TOKEN_CHANNEL', u'HIDDEN']
    modeNames = ['DEFAULT_MODE']
    literalNames = ['<INVALID>', "';'", "'('", "')'", "','", "'.'", "'/*+'", "'*/'", "'->'", "'['", "']'", "':'", "'ADD'", "'AFTER'", "'ALL'", "'ALTER'", "'ANALYZE'", "'AND'", "'ANTI'", "'ANY'", "'ARCHIVE'", "'ARRAY'", "'AS'", "'ASC'", "'AT'", "'AUTHORIZATION'", "'BETWEEN'", "'BOTH'", "'BUCKET'", "'BUCKETS'", "'BY'", "'CACHE'", "'CASCADE'", "'CASE'", "'CAST'", "'CHANGE'", "'CHECK'", "'CLEAR'", "'CLUSTER'", "'CLUSTERED'", "'CODEGEN'", "'COLLATE'", "'COLLECTION'", "'COLUMN'", "'COLUMNS'", "'COMMENT'", "'COMMIT'", "'COMPACT'", "'COMPACTIONS'", "'COMPUTE'", "'CONCATENATE'", "'CONSTRAINT'", "'COST'", "'CREATE'", "'CROSS'", "'CUBE'", "'CURRENT'", "'CURRENT_DATE'", "'CURRENT_TIME'", "'CURRENT_TIMESTAMP'", "'CURRENT_USER'", "'DATA'", "'DATABASE'", "'DAY'", "'DBPROPERTIES'", "'DEFINED'", "'DELETE'", "'DELIMITED'", "'DESC'", "'DESCRIBE'", "'DFS'", "'DIRECTORIES'", "'DIRECTORY'", "'DISTINCT'", "'DISTRIBUTE'", "'DROP'", "'ELSE'", "'END'", "'ESCAPE'", "'ESCAPED'", "'EXCEPT'", "'EXCHANGE'", "'EXISTS'", "'EXPLAIN'", "'EXPORT'", "'EXTENDED'", "'EXTERNAL'", "'EXTRACT'", "'FALSE'", "'FETCH'", "'FIELDS'", "'FILTER'", "'FILEFORMAT'", "'FIRST'", "'FOLLOWING'", "'FOR'", "'FOREIGN'", "'FORMAT'", "'FORMATTED'", "'FROM'", "'FULL'", "'FUNCTION'", "'FUNCTIONS'", "'GLOBAL'", "'GRANT'", "'GROUP'", "'GROUPING'", "'HAVING'", "'HOUR'", "'IF'", "'IGNORE'", "'IMPORT'", "'IN'", "'INDEX'", "'INDEXES'", "'INNER'", "'INPATH'", "'INPUTFORMAT'", "'INSERT'", "'INTERSECT'", "'INTERVAL'", "'INTO'", "'IS'", "'ITEMS'", "'JOIN'", "'KEYS'", "'LAST'", "'LATERAL'", "'LAZY'", "'LEADING'", "'LEFT'", "'LIKE'", "'LIMIT'", "'LINES'", "'LIST'", "'LOAD'", "'LOCAL'", "'LOCATION'", "'LOCK'", "'LOCKS'", "'LOGICAL'", "'MACRO'", "'MAP'", "'MATCHED'", "'MERGE'", "'MINUTE'", "'MONTH'", "'MSCK'", "'NAMESPACE'", "'NAMESPACES'", "'NATURAL'", "'NO'", "'NULL'", "'NULLS'", "'OF'", "'ON'", "'ONLY'", "'OPTION'", "'OPTIONS'", "'OR'", "'ORDER'", "'OUT'", "'OUTER'", "'OUTPUTFORMAT'", "'OVER'", "'OVERLAPS'", "'OVERLAY'", "'OVERWRITE'", "'PARTITION'", "'PARTITIONED'", "'PARTITIONS'", "'PERCENT'", "'PIVOT'", "'PLACING'", "'POSITION'", "'PRECEDING'", "'PRIMARY'", "'PRINCIPALS'", "'PROPERTIES'", "'PURGE'", "'QUERY'", "'RANGE'", "'RECORDREADER'", "'RECORDWRITER'", "'RECOVER'", "'REDUCE'", "'REFERENCES'", "'REFRESH'", "'RENAME'", "'REPAIR'", "'REPLACE'", "'RESET'", "'RESTRICT'", "'REVOKE'", "'RIGHT'", "'ROLE'", "'ROLES'", "'ROLLBACK'", "'ROLLUP'", "'ROW'", "'ROWS'", "'SCHEMA'", "'SECOND'", "'SELECT'", "'SEMI'", "'SEPARATED'", "'SERDE'", "'SERDEPROPERTIES'", "'SESSION_USER'", "'SET'", "'MINUS'", "'SETS'", "'SHOW'", "'SKEWED'", "'SOME'", "'SORT'", "'SORTED'", "'START'", "'STATISTICS'", "'STORED'", "'STRATIFY'", "'STRUCT'", "'SUBSTR'", "'SUBSTRING'", "'TABLE'", "'TABLES'", "'TABLESAMPLE'", "'TBLPROPERTIES'", "'TERMINATED'", "'THEN'", "'TO'", "'TOUCH'", "'TRAILING'", "'TRANSACTION'", "'TRANSACTIONS'", "'TRANSFORM'", "'TRIM'", "'TRUE'", "'TRUNCATE'", "'TYPE'", "'UNARCHIVE'", "'UNBOUNDED'", "'UNCACHE'", "'UNION'", "'UNIQUE'", "'UNKNOWN'", "'UNLOCK'", "'UNSET'", "'UPDATE'", "'USE'", "'USER'", "'USING'", "'VALUES'", "'VIEW'", "'VIEWS'", "'WHEN'", "'WHERE'", "'WINDOW'", "'WITH'", "'YEAR'", "'<=>'", "'<>'", "'!='", "'<'", "'>'", "'+'", "'-'", "'*'", "'/'", "'%'", "'DIV'", "'~'", "'&'", "'|'", "'||'", "'^'"]
    symbolicNames = ['<INVALID>', 'ADD', 'AFTER', 'ALL', 'ALTER', 'ANALYZE', 'AND', 'ANTI', 'ANY', 'ARCHIVE', 'ARRAY', 'AS', 'ASC', 'AT', 'AUTHORIZATION', 'BETWEEN', 'BOTH', 'BUCKET', 'BUCKETS', 'BY', 'CACHE', 'CASCADE', 'CASE', 'CAST', 'CHANGE', 'CHECK', 'CLEAR', 'CLUSTER', 'CLUSTERED', 'CODEGEN', 'COLLATE', 'COLLECTION', 'COLUMN', 'COLUMNS', 'COMMENT', 'COMMIT', 'COMPACT', 'COMPACTIONS', 'COMPUTE', 'CONCATENATE', 'CONSTRAINT', 'COST', 'CREATE', 'CROSS', 'CUBE', 'CURRENT', 'CURRENT_DATE', 'CURRENT_TIME', 'CURRENT_TIMESTAMP', 'CURRENT_USER', 'DATA', 'DATABASE', 'DATABASES', 'DAY', 'DBPROPERTIES', 'DEFINED', 'DELETE', 'DELIMITED', 'DESC', 'DESCRIBE', 'DFS', 'DIRECTORIES', 'DIRECTORY', 'DISTINCT', 'DISTRIBUTE', 'DROP', 'ELSE', 'END', 'ESCAPE', 'ESCAPED', 'EXCEPT', 'EXCHANGE', 'EXISTS', 'EXPLAIN', 'EXPORT', 'EXTENDED', 'EXTERNAL', 'EXTRACT', 'FALSE', 'FETCH', 'FIELDS', 'FILTER', 'FILEFORMAT', 'FIRST', 'FOLLOWING', 'FOR', 'FOREIGN', 'FORMAT', 'FORMATTED', 'FROM', 'FULL', 'FUNCTION', 'FUNCTIONS', 'GLOBAL', 'GRANT', 'GROUP', 'GROUPING', 'HAVING', 'HOUR', 'IF', 'IGNORE', 'IMPORT', 'IN', 'INDEX', 'INDEXES', 'INNER', 'INPATH', 'INPUTFORMAT', 'INSERT', 'INTERSECT', 'INTERVAL', 'INTO', 'IS', 'ITEMS', 'JOIN', 'KEYS', 'LAST', 'LATERAL', 'LAZY', 'LEADING', 'LEFT', 'LIKE', 'LIMIT', 'LINES', 'LIST', 'LOAD', 'LOCAL', 'LOCATION', 'LOCK', 'LOCKS', 'LOGICAL', 'MACRO', 'MAP', 'MATCHED', 'MERGE', 'MINUTE', 'MONTH', 'MSCK', 'NAMESPACE', 'NAMESPACES', 'NATURAL', 'NO', 'NOT', 'NULL', 'NULLS', 'OF', 'ON', 'ONLY', 'OPTION', 'OPTIONS', 'OR', 'ORDER', 'OUT', 'OUTER', 'OUTPUTFORMAT', 'OVER', 'OVERLAPS', 'OVERLAY', 'OVERWRITE', 'PARTITION', 'PARTITIONED', 'PARTITIONS', 'PERCENTLIT', 'PIVOT', 'PLACING', 'POSITION', 'PRECEDING', 'PRIMARY', 'PRINCIPALS', 'PROPERTIES', 'PURGE', 'QUERY', 'RANGE', 'RECORDREADER', 'RECORDWRITER', 'RECOVER', 'REDUCE', 'REFERENCES', 'REFRESH', 'RENAME', 'REPAIR', 'REPLACE', 'RESET', 'RESTRICT', 'REVOKE', 'RIGHT', 'RLIKE', 'ROLE', 'ROLES', 'ROLLBACK', 'ROLLUP', 'ROW', 'ROWS', 'SCHEMA', 'SECOND', 'SELECT', 'SEMI', 'SEPARATED', 'SERDE', 'SERDEPROPERTIES', 'SESSION_USER', 'SET', 'SETMINUS', 'SETS', 'SHOW', 'SKEWED', 'SOME', 'SORT', 'SORTED', 'START', 'STATISTICS', 'STORED', 'STRATIFY', 'STRUCT', 'SUBSTR', 'SUBSTRING', 'TABLE', 'TABLES', 'TABLESAMPLE', 'TBLPROPERTIES', 'TEMPORARY', 'TERMINATED', 'THEN', 'TO', 'TOUCH', 'TRAILING', 'TRANSACTION', 'TRANSACTIONS', 'TRANSFORM', 'TRIM', 'TRUE', 'TRUNCATE', 'TYPE', 'UNARCHIVE', 'UNBOUNDED', 'UNCACHE', 'UNION', 'UNIQUE', 'UNKNOWN', 'UNLOCK', 'UNSET', 'UPDATE', 'USE', 'USER', 'USING', 'VALUES', 'VIEW', 'VIEWS', 'WHEN', 'WHERE', 'WINDOW', 'WITH', 'YEAR', 'EQ', 'NSEQ', 'NEQ', 'NEQJ', 'LT', 'LTE', 'GT', 'GTE', 'PLUS', 'MINUS', 'ASTERISK', 'SLASH', 'PERCENT', 'DIV', 'TILDE', 'AMPERSAND', 'PIPE', 'CONCAT_PIPE', 'HAT', 'STRING', 'BIGINT_LITERAL', 'SMALLINT_LITERAL', 'TINYINT_LITERAL', 'INTEGER_VALUE', 'EXPONENT_VALUE', 'DECIMAL_VALUE', 'DOUBLE_LITERAL', 'BIGDECIMAL_LITERAL', 'IDENTIFIER', 'BACKQUOTED_IDENTIFIER', 'SIMPLE_COMMENT', 'BRACKETED_COMMENT', 'WS', 'UNRECOGNIZED']
    ruleNames = ['T__0', 'T__1', 'T__2', 'T__3', 'T__4', 'T__5', 'T__6', 'T__7', 'T__8', 'T__9', 'T__10', 'ADD', 'AFTER', 'ALL', 'ALTER', 'ANALYZE', 'AND', 'ANTI', 'ANY', 'ARCHIVE', 'ARRAY', 'AS', 'ASC', 'AT', 'AUTHORIZATION', 'BETWEEN', 'BOTH', 'BUCKET', 'BUCKETS', 'BY', 'CACHE', 'CASCADE', 'CASE', 'CAST', 'CHANGE', 'CHECK', 'CLEAR', 'CLUSTER', 'CLUSTERED', 'CODEGEN', 'COLLATE', 'COLLECTION', 'COLUMN', 'COLUMNS', 'COMMENT', 'COMMIT', 'COMPACT', 'COMPACTIONS', 'COMPUTE', 'CONCATENATE', 'CONSTRAINT', 'COST', 'CREATE', 'CROSS', 'CUBE', 'CURRENT', 'CURRENT_DATE', 'CURRENT_TIME', 'CURRENT_TIMESTAMP', 'CURRENT_USER', 'DATA', 'DATABASE', 'DATABASES', 'DAY', 'DBPROPERTIES', 'DEFINED', 'DELETE', 'DELIMITED', 'DESC', 'DESCRIBE', 'DFS', 'DIRECTORIES', 'DIRECTORY', 'DISTINCT', 'DISTRIBUTE', 'DROP', 'ELSE', 'END', 'ESCAPE', 'ESCAPED', 'EXCEPT', 'EXCHANGE', 'EXISTS', 'EXPLAIN', 'EXPORT', 'EXTENDED', 'EXTERNAL', 'EXTRACT', 'FALSE', 'FETCH', 'FIELDS', 'FILTER', 'FILEFORMAT', 'FIRST', 'FOLLOWING', 'FOR', 'FOREIGN', 'FORMAT', 'FORMATTED', 'FROM', 'FULL', 'FUNCTION', 'FUNCTIONS', 'GLOBAL', 'GRANT', 'GROUP', 'GROUPING', 'HAVING', 'HOUR', 'IF', 'IGNORE', 'IMPORT', 'IN', 'INDEX', 'INDEXES', 'INNER', 'INPATH', 'INPUTFORMAT', 'INSERT', 'INTERSECT', 'INTERVAL', 'INTO', 'IS', 'ITEMS', 'JOIN', 'KEYS', 'LAST', 'LATERAL', 'LAZY', 'LEADING', 'LEFT', 'LIKE', 'LIMIT', 'LINES', 'LIST', 'LOAD', 'LOCAL', 'LOCATION', 'LOCK', 'LOCKS', 'LOGICAL', 'MACRO', 'MAP', 'MATCHED', 'MERGE', 'MINUTE', 'MONTH', 'MSCK', 'NAMESPACE', 'NAMESPACES', 'NATURAL', 'NO', 'NOT', 'NULL', 'NULLS', 'OF', 'ON', 'ONLY', 'OPTION', 'OPTIONS', 'OR', 'ORDER', 'OUT', 'OUTER', 'OUTPUTFORMAT', 'OVER', 'OVERLAPS', 'OVERLAY', 'OVERWRITE', 'PARTITION', 'PARTITIONED', 'PARTITIONS', 'PERCENTLIT', 'PIVOT', 'PLACING', 'POSITION', 'PRECEDING', 'PRIMARY', 'PRINCIPALS', 'PROPERTIES', 'PURGE', 'QUERY', 'RANGE', 'RECORDREADER', 'RECORDWRITER', 'RECOVER', 'REDUCE', 'REFERENCES', 'REFRESH', 'RENAME', 'REPAIR', 'REPLACE', 'RESET', 'RESTRICT', 'REVOKE', 'RIGHT', 'RLIKE', 'ROLE', 'ROLES', 'ROLLBACK', 'ROLLUP', 'ROW', 'ROWS', 'SCHEMA', 'SECOND', 'SELECT', 'SEMI', 'SEPARATED', 'SERDE', 'SERDEPROPERTIES', 'SESSION_USER', 'SET', 'SETMINUS', 'SETS', 'SHOW', 'SKEWED', 'SOME', 'SORT', 'SORTED', 'START', 'STATISTICS', 'STORED', 'STRATIFY', 'STRUCT', 'SUBSTR', 'SUBSTRING', 'TABLE', 'TABLES', 'TABLESAMPLE', 'TBLPROPERTIES', 'TEMPORARY', 'TERMINATED', 'THEN', 'TO', 'TOUCH', 'TRAILING', 'TRANSACTION', 'TRANSACTIONS', 'TRANSFORM', 'TRIM', 'TRUE', 'TRUNCATE', 'TYPE', 'UNARCHIVE', 'UNBOUNDED', 'UNCACHE', 'UNION', 'UNIQUE', 'UNKNOWN', 'UNLOCK', 'UNSET', 'UPDATE', 'USE', 'USER', 'USING', 'VALUES', 'VIEW', 'VIEWS', 'WHEN', 'WHERE', 'WINDOW', 'WITH', 'YEAR', 'EQ', 'NSEQ', 'NEQ', 'NEQJ', 'LT', 'LTE', 'GT', 'GTE', 'PLUS', 'MINUS', 'ASTERISK', 'SLASH', 'PERCENT', 'DIV', 'TILDE', 'AMPERSAND', 'PIPE', 'CONCAT_PIPE', 'HAT', 'STRING', 'BIGINT_LITERAL', 'SMALLINT_LITERAL', 'TINYINT_LITERAL', 'INTEGER_VALUE', 'EXPONENT_VALUE', 'DECIMAL_VALUE', 'DOUBLE_LITERAL', 'BIGDECIMAL_LITERAL', 'IDENTIFIER', 'BACKQUOTED_IDENTIFIER', 'DECIMAL_DIGITS', 'EXPONENT', 'DIGIT', 'LETTER', 'SIMPLE_COMMENT', 'BRACKETED_COMMENT', 'WS', 'UNRECOGNIZED']
    grammarFileName = 'sql.g4'

    def __init__(self, input=None, output: TextIO=sys.stdout):
        super().__init__(input, output)
        self.checkVersion('4.11.1')
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None

    @property
    def legacy_setops_precedence_enbled(self):
        return False

    @property
    def legacy_exponent_literal_as_decimal_enabled(self):
        return False

    @property
    def isValidDecimal(self):
        return True
        nextChar = self._input.LA(1)
        if nextChar >= 'A' and nextChar <= 'Z' or (nextChar >= '0' and nextChar <= '9') or nextChar == '_':
            return False
        else:
            return True

    @property
    def SQL_standard_keyword_behavior(self):
        if '_ansi_sql' in self.__dict__:
            return self._ansi_sql
        return False

    def isHint(self):
        return False
        nextChar = self._input.LA(1)
        if nextChar == '+':
            return True
        else:
            return False

    @property
    def allUpperCase(self):
        if '_all_upper_case' in self.__dict__:
            return self._all_upper_case
        return False

    def sempred(self, localctx: RuleContext, ruleIndex: int, predIndex: int):
        if self._predicates is None:
            preds = dict()
            preds[287] = self.EXPONENT_VALUE_sempred
            preds[288] = self.DECIMAL_VALUE_sempred
            preds[289] = self.DOUBLE_LITERAL_sempred
            preds[290] = self.BIGDECIMAL_LITERAL_sempred
            preds[296] = self.LETTER_sempred
            preds[298] = self.BRACKETED_COMMENT_sempred
            self._predicates = preds
        pred = self._predicates.get(ruleIndex, None)
        if pred is not None:
            return pred(localctx, predIndex)
        else:
            raise Exception('No registered predicate for:' + str(ruleIndex))

    def EXPONENT_VALUE_sempred(self, localctx: RuleContext, predIndex: int):
        if predIndex == 0:
            return self.isValidDecimal

    def DECIMAL_VALUE_sempred(self, localctx: RuleContext, predIndex: int):
        if predIndex == 1:
            return self.isValidDecimal

    def DOUBLE_LITERAL_sempred(self, localctx: RuleContext, predIndex: int):
        if predIndex == 2:
            return self.isValidDecimal

    def BIGDECIMAL_LITERAL_sempred(self, localctx: RuleContext, predIndex: int):
        if predIndex == 3:
            return self.isValidDecimal

    def LETTER_sempred(self, localctx: RuleContext, predIndex: int):
        if predIndex == 4:
            return not self.allUpperCase

    def BRACKETED_COMMENT_sempred(self, localctx: RuleContext, predIndex: int):
        if predIndex == 5:
            return not self.isHint()